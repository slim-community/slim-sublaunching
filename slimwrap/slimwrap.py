from pathlib import Path
import tempfile
from typing import Any
import numpy as np
import subprocess
import os


def check_slim_script(file: str) -> None:
    try:
        subprocess.run(
            ["slim", "-c", file],
            text=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise ValueError(f"SLiM model check failed:\n{e.stderr.strip()}")


def parse_ndarray(key: str, val: np.ndarray) -> str:
    kind = val.dtype.kind
    if kind == "f":
        as_fn = "asFloat"
        values = ",".join(map(str, val))
    elif kind == "i":
        as_fn = "asInteger"
        values = ",".join(map(str, val))
    elif kind == "b":
        as_fn = "asLogical"
        values = ",".join("T" if x else "F" for x in val)
    elif kind in {"U", "S"}:
        as_fn = "asString"
        values = ",".join(f"'{x}'" for x in val)
    else:
        raise ValueError(
            f"Unsupported array dtype for SLiM constant: key={key}, dtype={val.dtype}"
        )
    return f"{key}={as_fn}(c({values}))"


def parse_matrix(key: str, val: np.ndarray) -> str:
    dims = val.shape
    val = val.reshape(-1)
    kind = val.dtype.kind
    if kind == "f":
        as_fn = "asFloat"
        values = ",".join(map(str, val))
    elif kind == "i":
        as_fn = "asInteger"
        values = ",".join(map(str, val))
    elif kind == "b":
        as_fn = "asLogical"
        values = ",".join("T" if x else "F" for x in val)
    elif kind in {"U", "S"}:
        as_fn = "asString"
        values = ",".join(f"'{x}'" for x in val)
    else:
        raise ValueError(
            f"Unsupported array dtype for SLiM constant: key={key}, dtype={val.dtype}"
        )
    return (
        f"{key}=matrix({as_fn}(c({values})), nrow={dims[0]}, ncol={dims[1]}, byrow=T)"
    )


def parse_key_value(key: str, val: Any) -> str:
    try:
        array = np.asarray(val)
        if array.ndim == 0 and array.dtype.kind in {"i", "f", "b", "U", "S"}:
            return parse_ndarray(key, np.array([val]))
        if (
            array.ndim == 1
            and array.size > 0
            and array.dtype.kind in {"i", "f", "b", "U", "S"}
        ):
            return parse_ndarray(key, array)
        if (
            array.ndim == 2
            and array.size > 0
            and array.dtype.kind in {"i", "f", "b", "U", "S"}
        ):
            return parse_matrix(key, array)
        if array.ndim > 2:
            ValueError("Unsupported number of dimensions")
    except Exception:
        pass
    raise ValueError(
        f"Unsupported type for SLiM constant: key={key}, value={val} (type={type(val)})"
    )


class SLiMModel:
    def __init__(self, model_source=None, model_code=None):
        """
        Initialize a SLiMModel instance.

        This constructor allows you to create a SLiMModel object either from a file
        containing SLiM model code or directly from a string of SLiM model code.

        Parameters:
        - model_source (str or Path, optional): Path to a file containing the SLiM model code.
        - model_code (str, optional): A string containing the SLiM model code.

        Raises:
        - TypeError: If both model_source and model_code are provided, or if neither is provided.
        - TypeError: If model_source is not a string or Path object.
        - ValueError: If the SLiM model code fails validation.
        """
        if model_source is not None and model_code is not None:
            raise TypeError(
                "Either model_source or model_code can be provided, not both"
            )
        if model_source is None and model_code is None:
            raise TypeError("Either model_source or model_code must be provided")

        self._temp_file = tempfile.NamedTemporaryFile(
            delete=False, mode="w", encoding="utf-8"
        )
        self._temp_filepath = self._temp_file.name

        if model_code is not None:
            self._temp_file.write(model_code)
        elif isinstance(model_source, str):
            with open(model_source, "r", encoding="utf-8") as f:
                self._temp_file.write(f.read())
        elif isinstance(model_source, Path):
            with model_source.open("r", encoding="utf-8") as f:
                self._temp_file.write(f.read())
        else:
            raise TypeError("model_source must be a str or Path")

        self._temp_file.flush()
        self._temp_file.close()
        check_slim_script(self._temp_filepath)
        self.last_result = None  # Store last run result

    def run(self, seed=None, constants=None, check=True):
        """
        Execute the SLiM model.

        This method runs the SLiM model using the specified seed and constants.
        It validates the input, constructs the appropriate command, and executes
        the SLiM script.

        Parameters:
        - seed (int, optional): A seed value for the SLiM simulation. If not provided,
          a random seed will be generated.
        - constants (dict, optional): A dictionary of constants to pass to the SLiM
          model. Keys should be strings representing variable names, and values
          should be compatible with SLiM's expected data types.
        - check (bool, optional): If True, raises an exception if the SLiM process
          exits with a non-zero status. Defaults to True.

        Returns:
        - subprocess.CompletedProcess: The result of the SLiM process execution,
          containing information such as stdout, stderr, and the return code.

        Raises:
        - TypeError: If the constants argument is not a dictionary.
        - ValueError: If the seed is not an integer or if an unsupported constant
          type is provided.
        - subprocess.CalledProcessError: If the SLiM process fails and `check` is True.
        """
        if constants is None:
            constants = {}
        if not isinstance(constants, dict):
            raise TypeError("constants argument should be a dictionary")
        commands = ["slim"]
        if seed is None:
            # TO-DO: Check what's the actual supported range of seed values
            self.last_seed = np.random.randint(1, 2**32, 1)[0]
        else:
            try:
                self.last_seed = int(seed)
            except:
                raise ValueError("seed should be and integer")
        commands.extend(["-s", f"{self.last_seed}"])
        for key, value in constants.items():
            commands.extend(["-d", parse_key_value(key, value)])
        commands.append(self._temp_filepath)
        self._last_seed = seed
        try:
            self.last_result = subprocess.run(
                commands, capture_output=True, shell=False, check=check
            )
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running SLiM:\n{e.stderr.decode().strip()}")
            raise
        return self.last_result

    def _repr_html_(self):
        try:
            with open(self._temp_filepath, "r", encoding="utf-8") as f:
                source = f.read()
        except OSError:
            source = "Model source code not available."

        model_html = f"""
        <div style="font-family: Arial, sans-serif; border: 1px solid #ddd; padding: 10px;">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <img src="http://benhaller.com/slim/icon/SLiM_256.jpg" alt="SLiM Logo" style="width: 40px; height: 40px; margin-right: 10px;">
                <h3 style="margin: 0;">SLiM model</h3>
            </div>
            <button onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'none' ? 'block' : 'none';"
                    style="background-color: #007BFF; color: white; border: none; padding: 5px 10px; cursor: pointer;">
                Show source code
            </button>
            <pre style="display: none; background-color: #f8f9fa; padding: 10px; border: 1px solid #ddd; overflow-x: auto;">
{source}
            </pre>
        </div>
        """

        output_html = ""
        if self.last_result:
            exit_code = self.last_result.returncode
            last_seed = "N/A" if not hasattr(self, "_last_seed") else self._last_seed
            stdout = (
                self.last_result.stdout.decode("utf-8")
                if self.last_result.stdout
                else "No stdout available."
            )
            stderr = (
                self.last_result.stderr.decode("utf-8")
                if self.last_result.stderr
                else "No stderr available."
            )

            output_html = f"""
            <div style="font-family: Arial, sans-serif; border: 1px solid #ddd; padding: 10px; margin-top: 20px;">
                <h3 style="margin-top: 0;">Output</h3>
                <p><strong>Exit Code:</strong> {exit_code}</p>
                <p><strong>Last Seed:</strong> {last_seed}</p>
                <button onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'none' ? 'block' : 'none';"
                        style="background-color: #007BFF; color: white; border: none; padding: 5px 10px; cursor: pointer;">
                    Show stdout
                </button>
                <pre style="display: none; background-color: #f8f9fa; padding: 10px; border: 1px solid #ddd; overflow-x: auto;">
{stdout}
                </pre>
                <button onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'none' ? 'block' : 'none';"
                        style="background-color: #007BFF; color: white; border: none; padding: 5px 10px; cursor: pointer; margin-top: 10px;">
                    Show stderr
                </button>
                <pre style="display: none; background-color: #f8f9fa; padding: 10px; border: 1px solid #ddd; overflow-x: auto;">
{stderr}
                </pre>
            </div>
            """
        return model_html + output_html

    @property
    def stdout(self):
        return self.last_result.stdout if self.last_result else None

    @property
    def stderr(self):
        return self.last_result.stderr if self.last_result else None

    @property
    def returncode(self):
        return self.last_result.returncode if self.last_result else None

    def __del__(self):
        try:
            os.remove(self._temp_filepath)
        except Exception:
            pass
