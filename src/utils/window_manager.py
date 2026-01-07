import pygetwindow as gw
import time

class WindowManager:
    def __init__(self, window_title_keyword="scrcpy"):
        self.window_title_keyword = window_title_keyword
        self.window = None

    def find_window(self):
        """Finds the window containing the keyword in its title."""
        windows = gw.getAllTitles()
        target_title = None
        for title in windows:
            if self.window_title_keyword.lower() in title.lower():
                target_title = title
                break
        
        if target_title:
            try:
                self.window = gw.getWindowsWithTitle(target_title)[0]
                print(f"Found window: {target_title}")
                return True
            except IndexError:
                print(f"Error accessing window: {target_title}")
                return False
        else:
            print(f"Window with keyword '{self.window_title_keyword}' not found.")
            return False

    def select_window(self):
        """Lists all windows and asks user to select one."""
        windows = [t for t in gw.getAllTitles() if t.strip()]
        for i, title in enumerate(windows):
            print(f"{i}: {title}")
        
        try:
            choice = int(input("Select window ID: "))
            if 0 <= choice < len(windows):
                target_title = windows[choice]
                self.window = gw.getWindowsWithTitle(target_title)[0]
                print(f"Selected window: {target_title}")
                return True
            else:
                print("Invalid selection.")
                return False
        except ValueError:
            print("Invalid input.")
            return False

    def get_window_rect(self):
        """Returns the (left, top, width, height) of the window client area if possible, else full window."""
        if not self.window:
            return None
        # Note: gw.top, gw.left are usually the outer bounds including title bar.
        # For precise capturing, we might need to adjust, but for now we take the whole window.
        return {
            "top": self.window.top,
            "left": self.window.left,
            "width": self.window.width,
            "height": self.window.height
        }

    def activate_window(self):
        if self.window:
            try:
                self.window.activate()
            except Exception as e:
                print(f"Could not activate window: {e}")

if __name__ == "__main__":
    wm = WindowManager()
    if wm.find_window():
        print(wm.get_window_rect())
