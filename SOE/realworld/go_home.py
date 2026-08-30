#!/usr/bin/env python

import time
import argparse

# Utility methods
from flexiv_rdk.example_py.utility import quat2eulerZYX
from flexiv_rdk.example_py.utility import parse_pt_states
from flexiv_rdk.example_py.utility import list2str

# Import Flexiv RDK Python library
# fmt: off
import sys
import os
# sys.path.insert(0, "flexiv_rdk/lib_py")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "flexiv_rdk/lib_py"))
import flexivrdk
from get_ip import get_ip
# fmt: on


def print_description():
    """
    Print tutorial description.

    """
    print(
        "This tutorial executes several basic robot primitives (unit skills). For "
        "detailed documentation on all available primitives, please see [Flexiv "
        "Primitives](https://www.flexiv.com/primitives/)."
    )
    print()


def main():
    # Program Setup
    # ==============================================================================================
    # Parse arguments
    # argparser = argparse.ArgumentParser()
    # argparser.add_argument("robot_ip", help="IP address of the robot server")
    # argparser.add_argument("local_ip", help="IP address of this PC")
    # args = argparser.parse_args()

    robot_ip, local_ip = get_ip()
    print("ip:",robot_ip, local_ip)

    # Define alias
    log = flexivrdk.Log()
    mode = flexivrdk.Mode

    # Print description
    log.info("Tutorial description:")
    print_description()

    try:
        # RDK Initialization
        # ==========================================================================================
        # Instantiate robot interface
        robot = flexivrdk.Robot(robot_ip, local_ip)

        # Clear fault on robot server if any
        if robot.isFault():
            log.warn("Fault occurred on robot server, trying to clear ...")
            # Try to clear the fault
            robot.clearFault()
            time.sleep(2)
            # Check again
            if robot.isFault():
                log.error("Fault cannot be cleared, exiting ...")
                return
            log.info("Fault on robot server is cleared")

        # Enable the robot, make sure the E-stop is released before enabling
        log.info("Enabling robot ...")
        robot.enable()

        # Wait for the robot to become operational
        while not robot.isOperational():
            time.sleep(1)

        log.info("Robot is now operational")

        # Execute Primitives
        # ==========================================================================================
        # Switch to primitive execution mode
        robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)

        # (1) Go to home pose
        # ------------------------------------------------------------------------------------------
        # All parameters of the "Home" primitive are optional, thus we can skip the parameters and
        # the default values will be used
        log.info("Executing primitive: Home")

        # Send command to robot
        robot.executePrimitive("Home()")

        # Wait for the primitive to finish
        while robot.isBusy():
            time.sleep(1)

        gripper = flexivrdk.Gripper(robot)
        gripper.move(0.1, 0.1, 20)

        # Wait for the primitive to finish
        while robot.isBusy():
            time.sleep(1)

        # All done, stop robot and put into IDLE mode
        robot.stop()

    except Exception as e:
        # Print exception error message
        log.error(str(e))


if __name__ == "__main__":
    main()