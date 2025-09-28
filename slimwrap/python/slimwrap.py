from pathlib import Path
import tempfile
from typing import Any
import numpy as np
import subprocess
import os
import json


def check_slim_script(file: str) -> None:
    """
    Validates a SLiM script file by running the SLiM interpreter in check mode.

    Parameters:
        file (str): The path to the SLiM script file to be validated.

    Raises:
        ValueError: If the SLiM script fails validation, an error message
                    containing the SLiM interpreter's stderr output is raised.
    """
    try:
        subprocess.run(
            ["slim", "-c", file],
            text=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise ValueError(f"SLiM model check failed:\n{e.stderr.strip()}")


class SLiMModel:
    def __init__(self, model_source=None, model_code=None):
        """
        Initializes a SLiMModel instance.

        Parameters:
            model_source (str or Path, optional): Path to a SLiM script file.
                                                    Either this or `model_code` must be provided, but not both.
            model_code (str, optional): SLiM script code as a string.
                                            Either this or `model_source` must be provided, but not both.

        Raises:
            TypeError: If both `model_source` and `model_code` are provided, or if neither is provided.
            TypeError: If `model_source` is not a str or Path.
            ValueError: If the SLiM script fails validation.
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
        Executes the SLiM model with optional parameters and a random or specified seed.

        Parameters:
            seed (int, optional): A specific seed value for the SLiM simulation. If not provided, a random seed is generated.
            constants (dict, optional): A dictionary of constants to pass to the SLiM model. Keys are variable names, and values
                                         can be scalars or numpy arrays (1D or 2D).
            check (bool, optional): If True, raises an exception if the SLiM process exits with a non-zero return code.

        Raises:
            TypeError: If `constants` is not a dictionary.
            ValueError: If `seed` is not an integer or if a numpy array in `constants` has more than 2 dimensions.
            subprocess.CalledProcessError: If the SLiM process fails and `check` is True.

        Notes:
            - If `constants` contains numpy arrays, they are converted to SLiM-compatible matrix representations. Additionally, a `Dictionary` named as 'SLIM_WRAP_PARAMS' is defined (which can be useful to further include in a tree-sequence).
            - Temporary files are used to pass constants to the SLiM model and are cleaned up after execution.
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
        # Parse optional arguments
        if constants:
            with tempfile.NamedTemporaryFile(
                delete=False, mode="w", encoding="utf-8"
            ) as param_file:
                y = constants.copy()
                for key, value in y.items():
                    if isinstance(value, np.ndarray):
                        y[key] = value.flatten().tolist()
                json.dump(y, param_file)
                param_file.flush()
                commands.extend(
                    [
                        "-d",
                        f"SLIM_WRAP_PARAMS = Dictionary(readFile('{param_file.name}'));",
                    ]
                )
                for key in constants.keys():
                    # If numpy array, use some metaprogramming to encode matrix
                    # f"{key}=matrix({as_fn}(c({values})), nrow={dims[0]}, ncol={dims[1]}, byrow=T)"
                    value = constants.get(key)
                    if isinstance(value, np.ndarray):
                        if value.ndim == 1:
                            commands.extend(
                                ["-d", f"{key}=SLIM_WRAP_PARAMS.getValue('{key}');"]
                            )
                        elif value.ndim == 2:
                            dims = value.shape
                            commands.extend(
                                [
                                    "-d",
                                    f"{key}=matrix(SLIM_WRAP_PARAMS.getValue('{key}'), nrow={dims[0]}, ncol={dims[1]}, byrow=T);",
                                ]
                            )
                        else:
                            raise ValueError(
                                f"Unsupported array with ndim > 2 for key '{key}'"
                            )
                    else:
                        commands.extend(
                            ["-d", f"{key}=SLIM_WRAP_PARAMS.getValue('{key}');"]
                        )
        commands.append(self._temp_filepath)
        try:
            last_result = subprocess.run(
                commands, capture_output=True, shell=False, check=check
            )
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running SLiM:\n{e.stderr.decode().strip()}")
            raise
        finally:
            if constants:
                try:
                    os.remove(param_file.name)
                except Exception as e:
                    print(f"Warning: Could not delete temporary parameter file: {e}")
        return last_result

    def _repr_html_(self):
        try:
            with open(self._temp_filepath, "r", encoding="utf-8") as f:
                source = f.read()
        except OSError:
            source = "Model source code not available."

        return f"""
        <div style="font-family: Arial, sans-serif; border: 1px solid #ddd; padding: 10px;">
            <div style="margin-bottom: 10px;">
                <h3 style="margin: 0;">SLiM model</h3>
            </div>
            <button onclick="const codeBlock = this.nextElementSibling;
                             if (codeBlock.style.display === 'none') {{
                                 codeBlock.style.display = 'block';
                                 this.textContent = 'Hide Source Code';
                             }} else {{
                                 codeBlock.style.display = 'none';
                                 this.textContent = 'Show Source Code';
                             }}"
                    style="background-color: #007BFF; color: white; border: none; padding: 5px 10px; cursor: pointer;">
                Hide Source Code
            </button>
            <pre style="display: block; padding: 10px; border: 1px solid #ddd; overflow-x: auto; font-family: monospace; white-space: pre-wrap;">
    <code class="language-c">{source}</code>
            </pre>
        </div>

        <!-- Load highlight.js -->
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
        <script>hljs.highlightAll();</script>
        """

    def __del__(self):
        try:
            os.remove(self._temp_filepath)
        except Exception:
            pass
