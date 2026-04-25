import pytest
from supervisor_agent_groq import *

class TestSupervisorNode:
    def test_state_with_analysis_key(self, mocker):
        state = {"analysis": True, "task": "test task", "steps": 0}
        mocker.patch('supervisor_agent_groq.llm.invoke')
        result = supervisor_node(state)
        assert "code_analyzer has run" in result["prompt"]

    def test_state_with_tests_key(self, mocker):
        state = {"tests": True, "task": "test task", "steps": 0}
        mocker.patch('supervisor_agent_groq.llm.invoke')
        result = supervisor_node(state)
        assert "test_generator has run" in result["prompt"]

    def test_state_with_both_analysis_and_tests_keys(self, mocker):
        state = {"analysis": True, "tests": True, "task": "test task", "steps": 0}
        mocker.patch('supervisor_agent_groq.llm.invoke')
        result = supervisor_node(state)
        assert "code_analyzer has run" in result["prompt"]
        assert "test_generator has run" in result["prompt"]

    def test_state_with_neither_analysis_nor_tests_key(self, mocker):
        state = {"task": "test task", "steps": 0}
        mocker.patch('supervisor_agent_groq.llm.invoke')
        result = supervisor_node(state)
        assert "nothing done yet" in result["prompt"]

    def test_response_with_code_analyzer(self, mocker):
        state = {"task": "test task", "steps": 0}
        response = mocker.Mock()
        response.content = "code_analyzer"
        mocker.patch('supervisor_agent_groq.llm.invoke', return_value=response)
        result = supervisor_node(state)
        assert result["next"] == "code_analyzer"

    def test_response_with_test_generator(self, mocker):
        state = {"task": "test task", "steps": 0}
        response = mocker.Mock()
        response.content = "test_generator"
        mocker.patch('supervisor_agent_groq.llm.invoke', return_value=response)
        result = supervisor_node(state)
        assert result["next"] == "test_generator"

    def test_response_with_finish(self, mocker):
        state = {"task": "test task", "steps": 0}
        response = mocker.Mock()
        response.content = "FINISH"
        mocker.patch('supervisor_agent_groq.llm.invoke', return_value=response)
        result = supervisor_node(state)
        assert result["next"] == "FINISH"

    def test_response_with_none_of_the_expected_keywords(self, mocker):
        state = {"task": "test task", "steps": 0}
        response = mocker.Mock()
        response.content = "unexpected response"
        mocker.patch('supervisor_agent_groq.llm.invoke', return_value=response)
        result = supervisor_node(state)
        assert result["next"] == "FINISH"

    def test_state_with_empty_task_key(self, mocker):
        state = {"task": "", "steps": 0}
        mocker.patch('supervisor_agent_groq.llm.invoke')
        result = supervisor_node(state)
        assert result["prompt"].startswith("You coordinate two workers on a software testing task.")

    def test_state_with_missing_task_key(self, mocker):
        state = {"steps": 0}
        mocker.patch('supervisor_agent_groq.llm.invoke')
        with pytest.raises(KeyError):
            supervisor_node(state)

    def test_state_with_non_string_task_value(self, mocker):
        state = {"task": 123, "steps": 0}
        mocker.patch('supervisor_agent_groq.llm.invoke')
        result = supervisor_node(state)
        assert result["prompt"].startswith("You coordinate two workers on a software testing task.")

    def test_state_with_non_dict_value_for_steps(self, mocker):
        state = {"task": "test task", "steps": "not an integer"}
        mocker.patch('supervisor_agent_groq.llm.invoke')
        result = supervisor_node(state)
        assert result["steps"] == 1

    def test_state_with_negative_steps_value(self, mocker):
        state = {"task": "test task", "steps": -1}
        mocker.patch('supervisor_agent_groq.llm.invoke')
        result = supervisor_node(state)
        assert result["steps"] == 1

    def test_llm_invoke_throwing_an_exception(self, mocker):
        state = {"task": "test task", "steps": 0}
        mocker.patch('supervisor_agent_groq.llm.invoke', side_effect=Exception("Test exception"))
        with pytest.raises(Exception):
            supervisor_node(state)

    def test_response_being_none(self, mocker):
        state = {"task": "test task", "steps": 0}
        mocker.patch('supervisor_agent_groq.llm.invoke', return_value=None)
        with pytest.raises(AttributeError):
            supervisor_node(state)

    def test_empty_state_dict(self, mocker):
        state = {}
        mocker.patch('supervisor_agent_groq.llm.invoke')
        with pytest.raises(KeyError):
            supervisor_node(state)

    def test_state_is_none(self, mocker):
        state = None
        mocker.patch('supervisor_agent_groq.llm.invoke')
        with pytest.raises(AttributeError):
            supervisor_node(state)

class TestCodeAnalyzerNode:
    def test_valid_state_object(self, mocker):
        state = {'task': 'def test(): pass'}
        mocker.patch('llm.invoke', return_value=mocker.Mock(content='test case'))
        response = code_analyzer_node(state)
        assert 'analysis' in response

    def test_invalid_state_object(self, mocker):
        state = None
        with pytest.raises(TypeError):
            code_analyzer_node(state)

    def test_empty_state_object(self, mocker):
        state = {}
        with pytest.raises(KeyError):
            code_analyzer_node(state)

    def test_none_state_object(self, mocker):
        state = None
        with pytest.raises(TypeError):
            code_analyzer_node(state)

    def test_state_object_missing_task_key(self, mocker):
        state = {}
        with pytest.raises(KeyError):
            code_analyzer_node(state)

    def test_state_object_having_empty_task_value(self, mocker):
        state = {'task': ''}
        response = code_analyzer_node(state)
        assert 'analysis' in response

    def test_llm_invoke_succeeding(self, mocker):
        state = {'task': 'def test(): pass'}
        mocker.patch('llm.invoke', return_value=mocker.Mock(content='test case'))
        response = code_analyzer_node(state)
        assert 'analysis' in response

    def test_llm_invoke_failing(self, mocker):
        state = {'task': 'def test(): pass'}
        mocker.patch('llm.invoke', side_effect=Exception('Mocked exception'))
        with pytest.raises(Exception):
            code_analyzer_node(state)

    def test_response_content_being_empty(self, mocker):
        state = {'task': 'def test(): pass'}
        mocker.patch('llm.invoke', return_value=mocker.Mock(content=''))
        response = code_analyzer_node(state)
        assert response['analysis'] == ''

    def test_response_content_being_none(self, mocker):
        state = {'task': 'def test(): pass'}
        mocker.patch('llm.invoke', return_value=mocker.Mock(content=None))
        response = code_analyzer_node(state)
        assert response['analysis'] is None

    def test_system_message_content_being_empty(self, mocker):
        state = {'task': 'def test(): pass'}
        mocker.patch('llm.invoke', return_value=mocker.Mock(content=''))
        response = code_analyzer_node(state)
        assert 'analysis' in response

    def test_human_message_content_being_empty(self, mocker):
        state = {'task': ''}
        response = code_analyzer_node(state)
        assert 'analysis' in response

    def test_function_returning_expected_dict_format(self, mocker):
        state = {'task': 'def test(): pass'}
        mocker.patch('llm.invoke', return_value=mocker.Mock(content='test case'))
        response = code_analyzer_node(state)
        assert isinstance(response, dict)
        assert 'analysis' in response

    def test_function_handling_exceptions(self, mocker):
        state = {'task': 'def test(): pass'}
        mocker.patch('llm.invoke', side_effect=Exception('Mocked exception'))
        with pytest.raises(Exception):
            code_analyzer_node(state)

    def test_function_logging_messages_correctly(self, mocker, caplog):
        state = {'task': 'def test(): pass'}
        mocker.patch('llm.invoke', return_value=mocker.Mock(content='test case'))
        code_analyzer_node(state)
        assert '  [CodeAnalyzer]   done' in caplog.text

    def test_function_returning_dict_with_content(self, mocker):
        state = {'task': 'def test(): pass'}
        mocker.patch('llm.invoke', return_value=mocker.Mock(content='test case'))
        response = code_analyzer_node(state)
        assert response['analysis'] == 'test case'

def test_test_generator_node_with_valid_state_and_task():
    state = {"task": "valid task", "analysis": "valid analysis"}
    result = test_generator_node(state)
    assert isinstance(result, dict)
    assert "tests" in result

def test_test_generator_node_with_valid_state_and_no_task():
    state = {"analysis": "valid analysis"}
    result = test_generator_node(state)
    assert isinstance(result, dict)
    assert "tests" in result

def test_test_generator_node_with_valid_state_and_empty_task():
    state = {"task": "", "analysis": "valid analysis"}
    result = test_generator_node(state)
    assert isinstance(result, dict)
    assert "tests" in result

def test_test_generator_node_with_valid_state_and_analysis():
    state = {"task": "valid task", "analysis": "valid analysis"}
    result = test_generator_node(state)
    assert isinstance(result, dict)
    assert "tests" in result

def test_test_generator_node_with_valid_state_and_no_analysis():
    state = {"task": "valid task"}
    result = test_generator_node(state)
    assert isinstance(result, dict)
    assert "tests" in result

def test_test_generator_node_with_valid_state_and_empty_analysis():
    state = {"task": "valid task", "analysis": ""}
    result = test_generator_node(state)
    assert isinstance(result, dict)
    assert "tests" in result

def test_test_generator_node_with_invalid_state():
    state = "invalid state"
    with pytest.raises(AttributeError):
        test_generator_node(state)

def test_test_generator_node_with_missing_state():
    with pytest.raises(TypeError):
        test_generator_node()

def test_test_generator_node_with_none_state():
    state = None
    with pytest.raises(AttributeError):
        test_generator_node(state)

def test_test_generator_node_with_state_containing_non_string_task():
    state = {"task": 123, "analysis": "valid analysis"}
    with pytest.raises(TypeError):
        test_generator_node(state)

def test_test_generator_node_with_state_containing_non_string_analysis():
    state = {"task": "valid task", "analysis": 123}
    with pytest.raises(TypeError):
        test_generator_node(state)

def test_test_generator_node_with_state_containing_non_dict_analysis():
    state = {"task": "valid task", "analysis": ["invalid analysis"]}
    with pytest.raises(TypeError):
        test_generator_node(state)

def test_llm_invoke_call_with_valid_input():
    state = {"task": "valid task", "analysis": "valid analysis"}
    context = f"Function to test:\n{state['task']}\n\nRequired test cases:\n{state['analysis']}"
    llm.invoke([
        SystemMessage(content=(
            "You are a pytest expert. "
            "Output ONLY valid Python pytest code â€” no markdown fences, no prose. "
            "Use descriptive test function names. Include edge cases."
        )),
        HumanMessage(content=context),
    ])

def test_llm_invoke_call_with_invalid_input():
    state = {"task": "valid task", "analysis": "valid analysis"}
    context = f"Function to test:\n{state['task']}\n\nRequired test cases:\n{state['analysis']}"
    with pytest.raises(TypeError):
        llm.invoke([
            SystemMessage(content=(
                "You are a pytest expert. "
                "Output ONLY valid Python pytest code â€” no markdown fences, no prose. "
                "Use descriptive test function names. Include edge cases."
            )),
            HumanMessage(content=context),
            "invalid input"
        ])

def test_llm_invoke_call_with_missing_input():
    with pytest.raises(TypeError):
        llm.invoke()

def test_llm_invoke_call_with_none_input():
    with pytest.raises(TypeError):
        llm.invoke(None)

def test_test_generator_node_return_type_is_dict():
    state = {"task": "valid task", "analysis": "valid analysis"}
    result = test_generator_node(state)
    assert isinstance(result, dict)

def test_test_generator_node_return_dict_contains_tests_key():
    state = {"task": "valid task", "analysis": "valid analysis"}
    result = test_generator_node(state)
    assert "tests" in result

def test_test_generator_node_return_dict_contains_valid_pytest_code():
    state = {"task": "valid task", "analysis": "valid analysis"}
    result = test_generator_node(state)
    assert "def" in result["tests"]

def test_test_generator_node_handles_exceptions_from_llm_invoke_call():
    state = {"task": "valid task", "analysis": "valid analysis"}
    context = f"Function to test:\n{state['task']}\n\nRequired test cases:\n{state['analysis']}"
    with pytest.raises(Exception):
        llm.invoke([
            SystemMessage(content=(
                "You are a pytest expert. "
                "Output ONLY valid Python pytest code â€” no markdown fences, no prose. "
                "Use descriptive test function names. Include edge cases."
            )),
            HumanMessage(content=context),
        ])
        test_generator_node(state)

def test_test_generator_node_handles_empty_response_from_llm_invoke_call():
    state = {"task": "valid task", "analysis": "valid analysis"}
    context = f"Function to test:\n{state['task']}\n\nRequired test cases:\n{state['analysis']}"
    llm.invoke([
        SystemMessage(content=(
            "You are a pytest expert. "
            "Output ONLY valid Python pytest code â€” no markdown fences, no prose. "
            "Use descriptive test function names. Include edge cases."
        )),
        HumanMessage(content=context),
    ])
    result = test_generator_node(state)
    assert result["tests"] != ""

def test_test_generator_node_handles_response_with_invalid_pytest_code():
    state = {"task": "valid task", "analysis": "valid analysis"}
    context = f"Function to test:\n{state['task']}\n\nRequired test cases:\n{state['analysis']}"
    llm.invoke([
        SystemMessage(content=(
            "You are a pytest expert. "
            "Output ONLY valid Python pytest code â€” no markdown fences, no prose. "
            "Use descriptive test function names. Include edge cases."
        )),
        HumanMessage(content=context),
    ])
    result = test_generator_node(state)
    assert "def" in result["tests"]

class TestRouteFunction:
    def test_route_state_with_steps_greater_than_or_equal_to_six(self):
        state = {"steps": 6}
        assert route(state) == "end"

    def test_route_state_with_steps_less_than_six(self):
        state = {"steps": 5}
        assert route(state) == "end"

    def test_route_state_with_decision_code_analyzer(self):
        state = {"steps": 0, "next": "code_analyzer"}
        assert route(state) == "code_analyzer"

    def test_route_state_with_decision_test_generator(self):
        state = {"steps": 0, "next": "test_generator"}
        assert route(state) == "test_generator"

    def test_route_state_with_decision_other_than_code_analyzer_or_test_generator(self):
        state = {"steps": 0, "next": "other"}
        assert route(state) == "end"

    def test_route_state_with_missing_next_key(self):
        state = {"steps": 0}
        assert route(state) == "end"

    def test_route_state_with_missing_steps_key(self):
        state = {"next": "code_analyzer"}
        assert route(state) == "code_analyzer"

    def test_route_state_with_non_integer_value_for_steps(self):
        state = {"steps": "five"}
        with pytest.raises(Exception):
            route(state)

    def test_route_state_with_empty_state(self):
        state = {}
        assert route(state) == "end"

    def test_route_state_with_none_state(self):
        state = None
        with pytest.raises(AttributeError):
            route(state)

    def test_route_state_with_non_string_value_for_next_key(self):
        state = {"steps": 0, "next": 123}
        assert route(state) == "end"

