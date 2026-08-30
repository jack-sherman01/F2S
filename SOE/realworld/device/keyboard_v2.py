from easydict import EasyDict as edict
import numpy as np
from pynput.keyboard import Key, Listener

class Keyboard:

    def __init__(self, keymap=None):
        # self.keylist = []
        self.keymap = keymap if keymap is not None else {}
        self.state = edict({key: False for key in self.keymap.values()})
        # make a thread to listen to keyboard and register our callback functions
        self.listener = Listener(
            on_press=self.on_press, on_release=self.on_release)
        # start listening
        self.listener.start()

    def on_press(self, key):
        """
        Key handler for key presses.
        Args:
            key (str): key that was pressed
        """
        try:
            if key.char is not None:
                # self.keylist.append(key.char)
                if key.char in self.keymap:
                    self.state[self.keymap[key.char]] = True
        except AttributeError:
            key_name = str(key).replace("Key.", "")
            if key_name in self.keymap:
                self.state[self.keymap[key_name]] = True

    def on_release(self, key):
        """
        Key handler for key releases.
        Args:
            key (str): key that was pressed
        """
        pass


if __name__ == '__main__':
    import time
    device = Keyboard(keymap={
        's': 'start',
        'f': 'finish',
        'd': 'discard',
        'q': 'quit',
        'esc': 'escape',
    })
    while True:
        time.sleep(0.1)
        if device.state['start']:
            print("Start pressed")
            device.state['start'] = False
        if device.state['finish']:
            print("Finish pressed")
            device.state['finish'] = False
        if device.state['discard']:
            print("Discard pressed")
            device.state['discard'] = False
        if device.state['quit']:
            print("Quit pressed")
            device.state['quit'] = False
        if device.state['escape']:
            print("Escape pressed")
            device.state['escape'] = False
