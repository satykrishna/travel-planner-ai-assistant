import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

from planner import observability as obs


def test_trace_batches_hide_contents_but_preserve_usage(monkeypatch):
    monkeypatch.setenv('LANGSMITH_API_KEY','test-langsmith-secret-12345678')
    raw = {'id':'safe-run-id', 'name':'search_flights', 'run_type':'llm',
           'inputs':{'query':'PRIVATE TRIP QUERY'}, 'outputs':{'text':'PRIVATE ANSWER'},
           'events':[{'text':'PRIVATE EVENT'}], 'attachments':{'secret':'PRIVATE ATTACHMENT'},
           'serialized':{'prompt':'PRIVATE SYSTEM PROMPT'},
           'error':'Request failed: test-langsmith-secret-12345678 at private@example.test',
           'extra':{'runtime':{'secret':'PRIVATE RUNTIME'}, 'metadata':{
               'request_id':'safe-request-id','ls_model_name':'gpt-4o-mini','private_query':'PRIVATE QUERY',
               'usage_metadata':{'input_tokens':100,'output_tokens':20,'total_tokens':120,'unexpected':'PRIVATE USAGE'}}}}
    result = obs.safe_run_operations([raw])[0]
    assert result['inputs'] == {} and result['outputs'] == {}
    assert result['extra']['metadata']['usage_metadata'] == {'input_tokens':100,'output_tokens':20,'total_tokens':120}
    assert result['extra']['metadata']['ls_model_name'] == 'gpt-4o-mini'
    assert 'PRIVATE' not in str(result) and 'test-langsmith-secret-12345678' not in str(result)
    assert 'private@example.test' not in str(result)
    assert raw['inputs'] != {}  # Source payload is not mutated.


def test_no_key_disables_tracing_without_disabling_planning(monkeypatch):
    monkeypatch.delenv('LANGSMITH_API_KEY',raising=False)
    obs.get_client.cache_clear()
    with obs.observe_request() as observation:
        observation.record('{"event":"result"}')
        observation.finish({'model_calls':3,'search_calls':1})
        asyncio.run(observation.flush())
        assert observation.client is None and observation.outcome == 'success'
    obs.get_client.cache_clear()


def test_client_uses_privacy_filter_and_no_retries(monkeypatch):
    monkeypatch.setenv('LANGSMITH_TRACING','true')
    monkeypatch.setenv('LANGSMITH_API_KEY','fake-key')
    monkeypatch.setenv('LANGSMITH_ENDPOINT','https://api.smith.langchain.com')
    factory = Mock(return_value=object())
    monkeypatch.setattr(obs,'Client',factory)
    obs.get_client.cache_clear()
    assert obs.get_client() is not None
    kwargs = factory.call_args.kwargs
    assert kwargs['hide_inputs'] and kwargs['hide_outputs'] and kwargs['omit_traced_runtime_info']
    assert kwargs['process_buffered_run_ops'] is obs.safe_run_operations
    assert kwargs['retry_config'].total == 0
    obs.get_client.cache_clear()


def test_unsupported_endpoint_never_receives_key(monkeypatch):
    monkeypatch.setenv('LANGSMITH_TRACING','true')
    monkeypatch.setenv('LANGSMITH_API_KEY','fake-key')
    monkeypatch.setenv('LANGSMITH_ENDPOINT','https://attacker.test')
    factory = Mock()
    monkeypatch.setattr(obs,'Client',factory)
    obs.get_client.cache_clear()
    assert obs.get_client() is None
    factory.assert_not_called()
    obs.get_client.cache_clear()


def test_observation_reports_counts_and_flushes(monkeypatch):
    client = SimpleNamespace(flush=Mock())
    run = SimpleNamespace(end=Mock())
    observation = obs.Observation(client=client,run=run)
    observation.record('{"event":"error","data":{"message":"PRIVATE INPUT"}}')
    observation.finish({'model_calls':5,'search_calls':2})
    assert run.end.call_args.kwargs['metadata'] == {'outcome':'blocked_or_failed','model_calls':5,'search_calls':2}
    assert 'PRIVATE' not in str(run.end.call_args)
    asyncio.run(observation.flush())
    client.flush.assert_called_once_with(timeout=2)


def test_tracing_setup_failure_does_not_block_planning(monkeypatch):
    monkeypatch.setattr(obs,'get_client',lambda: object())
    @contextmanager
    def broken_trace(*args,**kwargs):
        raise RuntimeError('Offline')
        yield
    monkeypatch.setattr(obs,'trace',broken_trace)
    with obs.observe_request() as observation:
        assert observation.client is None
        observation.record('{"event":"result"}')
        assert observation.outcome == 'success'
