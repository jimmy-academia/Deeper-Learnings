

'''
Interactive Debugging Utility:

Enhances Python debugging by providing an interactive console for inspecting and navigating stack frames when exceptions occur or when manually invoked. This utility allows developers to examine and modify local and global variables in real-time, facilitating easier identification and resolution of issues.

- **Usage:**
    - **Enable Debug Mode:**
        - Call `debug_mode()` to activate custom exception handling. Upon an uncaught exception, an interactive console with frame navigation will be launched.
    - **Manual Inspection:**
        - Call `check()` within your code to capture the current local and global variables and open an interactive console for inspection.

- **Components:**
    - **Main Functions:**
        - `debug_mode()`: Sets up a custom exception hook to handle uncaught exceptions and initiate the interactive debugger.
        - `check()`: Manually captures the caller’s local and global variables and starts an interactive console for debugging.
        - `custom_excepthook()`: Custom exception handler that processes exceptions, prints tracebacks, and invokes `syscheck()` for interactive debugging.

    - **Internal Utilities:**
        - `syscheck()`: Restores the original exception hook, gathers user-defined stack frames from the last traceback, and launches the interactive console with frame navigation.
        - `FrameNavigator`: A class that manages navigation through collected stack frames, allowing users to move between frames and inspect variables.
            - Methods:
                - `next()`: Move to the next (younger) frame.
                - `prev()`: Move to the previous (older) frame.
                - `list()`: List all collected frames with indicators for the current frame.
        - `is_user_code()`: Determines if a stack frame originates from user-defined code by checking the file path.
        - `list_vars()`: Displays local variables in the current frame, excluding built-ins and callable objects.
'''

import sys
import code 
import inspect
import traceback

def check():
    current_frame = inspect.currentframe()
    caller_frame = current_frame.f_back
    caller_locals = caller_frame.f_locals
    caller_globals = caller_frame.f_globals
    for key in caller_globals:
        if key not in globals():
            globals()[key] = caller_globals[key]

    frame_info = inspect.getframeinfo(caller_frame)
    caller_file = frame_info.filename
    caller_line = frame_info.lineno

    print('### check function called...')
    print(f"Called from {caller_file}")
    print(f"--------->> at line {caller_line}")

    code.interact(local=dict(globals(), **caller_locals))

original_excepthook = sys.excepthook

def debug_mode():
    sys.excepthook = custom_excepthook

def custom_excepthook(exctype, value, tb):
    if exctype == KeyboardInterrupt:
        print("KeyboardInterrupt caught. Exiting cleanly.")
        sys.exit(0)
    else:
        traceback.print_exception(exctype, value, tb)
        sys.last_traceback = tb  # Save the last traceback to use in check()
        syscheck()

def syscheck():
    # Restore the original excepthook
    sys.excepthook = original_excepthook

    # Get the last traceback
    tb = sys.last_traceback
    user_frames = []

    # Collect all user frames
    while tb:
        frame = tb.tb_frame
        if is_user_code(frame):
            user_frames.append(frame)
        tb = tb.tb_next

    if not user_frames:
        print("No user frames found")
        return

    global interactive_locals
    interactive_locals = {}

    navigator = FrameNavigator(user_frames)

    # Interactive console with frame navigation
    banner = (
        "\n"
        "=== Interactive mode ===\n"
        "Use 'nv.next()' to go to the next frame, "
        "Use 'nv.prev()' to go to the previous frame.\n"
        "Use 'nv.list()' to list all frames.\n"
        "Use 'ls()' to list local variables in the current frame.\n"
        "Local variables of the current frame are accessible.\n"
    )

    def interact():
        navigator.update_interactive_locals()
        code.interact(banner, local=interactive_locals)

    interact()


class FrameNavigator:
    def __init__(self, frames):
        self.frames = frames
        self.current_frame_index = 0
        self.update_context(self.current_frame_index)

    def update_context(self, index):
        self.current_frame_index = index
        frame = self.frames[index]
        self.locals = frame.f_locals.copy()
        self.globals = frame.f_globals
        frame_info = inspect.getframeinfo(frame)
        self.filename = frame_info.filename
        self.lineno = frame_info.lineno
        print(f"Switched to frame {index}: {self.filename} at line {self.lineno}")
        # Update interactive console locals
        self.update_interactive_locals()

    def update_interactive_locals(self):
        # Update the locals dictionary in the interactive console
        interactive_locals.clear()
        interactive_locals.update(self.locals)
        interactive_locals.update({'nv': self, 'ls': list_vars})

    def next(self):
        if self.current_frame_index < len(self.frames) - 1:
            self.update_context(self.current_frame_index + 1)
        else:
            print("Already at the newest frame")

    def prev(self):
        if self.current_frame_index > 0:
            self.update_context(self.current_frame_index - 1)
        else:
            print("Already at the oldest frame")

    def list(self):
        print("Frames:")
        for i, frame in enumerate(self.frames):
            frame_info = inspect.getframeinfo(frame)
            prefix = "* " if i == self.current_frame_index else ""
            print(f"{prefix}frame {i}: {frame_info.filename} at line {frame_info.lineno}")


def is_user_code(frame):
    # Check if the frame is from user code by comparing the file path
    filename = frame.f_globals["__file__"]
    return not filename.startswith(sys.prefix)

def list_vars():
    print("Local variables in the current frame:")
    for var, val in interactive_locals.items():
        if not var.startswith("__") and not callable(val):
            print(f"{var}: {val}")

