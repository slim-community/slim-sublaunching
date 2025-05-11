import os
import pytest
import tempfile
import numpy as np
from slimwrap import SLiMModel, parse_key_value

MINIMAL_VALID_SCRIPT = """
initialize() {
    initializeMutationRate(1e-7);
    initializeMutationType("m1", 0.5, "f", 0.0);
    initializeGenomicElementType("g1", m1, 1.0);
    initializeGenomicElement(g1, 0, 99999);
    initializeRecombinationRate(1e-8);
}
1 early() {
    sim.addSubpop("p1", 500);
}
2000 late() { sim.outputFixedMutations(); }
"""

MINIMAL_INVALID_SCRIPT = """
initialize() {initializeMutationRate(
1 early() {sim.addSubpop("p1", 500);}
2000 late() { sim.outputFixedMutations(); }
"""


def test_model_init():
    "Test that we can initialize a SLiMModel from a valid file in memory"
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_file.write(MINIMAL_VALID_SCRIPT)
        temp_file_path = temp_file.name
    try:
        SLiMModel(model_source=temp_file_path)
    except Exception as e:
        pytest.fail(f"Model run failed: {e}")
    finally:
        os.remove(temp_file_path)


def test_fail_invalid_script():
    "Test that we produce an informative error if there's a syntax error"
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_file.write(MINIMAL_INVALID_SCRIPT)
        temp_file_path = temp_file.name
    with pytest.raises(Exception):
        SLiMModel(model_source=temp_file_path)
    os.remove(temp_file_path)


def test_temp_model_init():
    "Test that we can initialize a model with a temporary file"
    SLiMModel(model_code=MINIMAL_VALID_SCRIPT)
    with pytest.raises(Exception):
        SLiMModel(model_code=MINIMAL_INVALID_SCRIPT)


def test_run_model():
    """Test that we can run a simple model:"""
    model = SLiMModel(model_code=MINIMAL_VALID_SCRIPT)
    model.run()
    model.run(seed=1000)

    with pytest.raises(Exception):
        model.run(seed="a")
    model.run(constants={"A": 1e-8})


def test_arguments_parsing():
    "Test that we support basic primitives and basic arrays"
    # Basic primitives
    assert parse_key_value("key", "val") == "key=asString(c('val'))"
    assert parse_key_value("key", 1.2) == "key=asFloat(c(1.2))"
    assert parse_key_value("key", 1) == "key=asInteger(c(1))"
    assert parse_key_value("key", True) == "key=asLogical(c(T))"
    assert parse_key_value("key", False) == "key=asLogical(c(F))"
    # Basic arrays
    assert parse_key_value("key", [1.2, 1.3]) == "key=asFloat(c(1.2,1.3))"
    assert parse_key_value("key", np.array([1.2, 1.3])) == "key=asFloat(c(1.2,1.3))"
    assert parse_key_value("key", [1, 2]) == "key=asInteger(c(1,2))"
    assert parse_key_value("key", [True, False]) == "key=asLogical(c(T,F))"
    assert parse_key_value("key", ["a", "b"]) == "key=asString(c('a','b'))"
    # Basic matrix
    mat = np.array([[1, 1], [2, 2]])
    assert (
        parse_key_value("key", mat)
        == "key=matrix(asInteger(c(1,1,2,2)), nrow=2, ncol=2, byrow=T)"
    )


def test_arguments_definition():
    "Test that we are defining the arguments properly inside SLiM"
    model = """
    initialize() {
	initializeMutationRate(1e-7);
	initializeMutationType("m1", 0.5, "f", 0.0);
	initializeGenomicElementType("g1", m1, 1.0);
	initializeGenomicElement(g1, 0, 99999);
	initializeRecombinationRate(1e-8);
    }
    1 early() {
        assert(twoFloat==2.0);
        assert(twoInt==2);
        assert(yes==T);
        assert(no==F);
        assert(all(bool_vector==c(T, F, T)));
        assert(sum(float_vector - c(2.3, 1.1, 1.0)) < 1e-10);
        assert(sum(int_vector - c(1, 2, 3)) < 1e-10);
        assert(all(str_vector==c("1", "True", "aa")));
        // Check matrix
        assert(all(matrix[0, 0] == 1));
        assert(all(matrix[0, 1] == 2));
        assert(all(matrix[1, 0] == 3));
        assert(all(matrix[1, 1] == 4));
    }
    """
    model = SLiMModel(model_code=model)
    params = {
        "twoFloat": 2.0,
        "twoInt": 2,
        "yes": True,
        "no": False,
        "bool_vector": [True, False, True],
        "float_vector": [2.3, 1.1, 1.0],
        "int_vector": [1, 2, 3],
        "str_vector": ["1", "True", "aa"],
        "matrix": np.array([[1, 2], [3, 4]]),
    }
    model.run(constants=params)
