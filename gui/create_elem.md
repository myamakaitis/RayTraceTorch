### **1. Core Architectural Requirements**

* **Introspection-First:** The GUI must inspect the `__init__` method of the selected class to determine fields.
* **Recursion:** Support nested classes (e.g., an `Element` containing a `RayTransform` object). Arguments that are classes must generate nested `QGroupBox` containers.
* **Bottom-Up Instantiation:** When "OK" is clicked, the system must build the inner-most objects first (e.g., the `RayTransform`), then pass them as arguments to the outer object (the `Element`).
* **Inheritance Scanning:** The class selector must populate by recursively (iteratively) finding all subclasses of a base `Element` class.

### **2. Type Handling & Custom Widgets**

The introspection logic must handle complex type hints used in scientific computing.

* **Custom Aliases:**
* `Vector3`: Defined as `Union[torch.Tensor, List[float], Tuple[float, ...]]`. Must render as **3 side-by-side QLineEdits** (float).
* `Bool3`: Defined as `Union[torch.Tensor, List[bool], Tuple[bool, ...]]`. Must render as **3 side-by-side QCheckBoxes**.


* **PyTorch Types:** `torch.dtype` must render as a dropdown (`float32` / `float64`).
* **Primitives:** Standard `int`, `float`, `str` render as `QLineEdit`. `bool` renders as `QCheckBox`.
* **Unwrapping Strategy:** The analyzer must strip `Optional[...]` and `Union[..., None]` wrappers to identify the underlying intent (e.g., `Optional[Vector3]`  treat as `Vector3`).

### **3. UI Logic & formatting**

* **Gradient Pairing:**
* **Requirement:** Many parameters (e.g., `c1`) have a corresponding boolean flag for differentiation (e.g., `c1_grad`).
* **Behavior:** The GUI must detect these pairs and render the gradient checkbox **inline** (next to the main parameter widget) rather than on a new line.
* **Abbreviation Handling:** Must support specific naming mismatches, specifically:
* `translation`  `trans_grad`
* `rotation`  `rot_grad`




* **Optional Sub-Objects:**
* If a nested class argument defaults to `None` (e.g., `transform: RayTransform = None`), the generated `QGroupBox` must be **Checkable**.
* **Unchecked:** Passes `None` to the constructor.
* **Checked:** Instantiates the object using the sub-form values.



### **4. Stability & Crash Prevention**

* **Iterative Subclassing:** Use an iterative queue method for `get_subclasses` to avoid recursion errors on Windows.
* **Safe Layout Reset:** Use `widget.deleteLater()` on the container widget rather than trying to delete a layout directly (prevents `0xC0000409` stack buffer overruns).
* **Recursion Depth Limit:** Hard cap recursion depth (e.g., 5 levels) to prevent infinite loops if classes reference each other.
* **Safe Evaluation:** Use `ast.literal_eval` for parsing text inputs, not `eval()`.
* **Default Value Safety:** Custom widgets (`Vector3Input`) must handle scalar or `None` defaults by expanding them to lists (e.g., `0.0`  `[0.0, 0.0, 0.0]`) to prevent `len()` crashes.

### **5. Key Helpers (Reference for AI)**

Any modification must preserve these helper functions which solve the complexity of `typing` inspection:

```python
def analyze_type(p_type):
    """
    Unpacks nested Unions/Optionals to find if torch.Tensor, List[bool], etc. 
    are present, returning tags like 'VEC3', 'BOOL3', 'CLASS', 'PRIMITIVE'.
    """

def find_gradient_partner(name, all_params):
    """
    Matches 'c1' to 'c1_grad' and handles 'translation' -> 'trans_grad' mapping.
    """

```