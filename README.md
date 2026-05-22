# Hornsby

Hornsby is an experimental educational robot for learning programming, computer vision, embedded systems, and physical AI through a real moving machine.

The goal is to make robotics feel practical, low-cost, and approachable: a small robot that can be built, programmed, observed, improved, and used as a platform for learning how software interacts with the physical world.

## Purpose

Hornsby is designed as a hands-on practice robot for:

- programming fundamentals through real behavior;
- computer vision experiments with cameras and sensors;
- physical AI, where models must act in the real world;
- robotics control, feedback loops, and navigation;
- hardware learning with motors, batteries, boards, sensors, and mechanical constraints;
- student-friendly experiments that are simple enough to understand but real enough to matter.

This project is not only about making a robot move. It is about creating a learning platform where code produces visible physical results, mistakes can be observed, and every improvement teaches something concrete.

## Initial Direction

The first version should focus on a small, reliable base that can:

- move forward, backward, and turn;
- expose simple motor-control commands;
- support basic visual feedback;
- run repeatable programming exercises;
- provide a path toward computer vision and autonomous behavior;
- remain easy to build, repair, document, and explain.

The robot should stay simple at the beginning. A clean foundation will make it easier to add sensors, vision, autonomy, and AI behavior later.

## Current Design Files

The repository includes the first 3D/mechanical reference files in [`3D/`](3D/):

- [`robot_one_dollar_board.f3z`](3D/robot_one_dollar_board.f3z): initial Autodesk Fusion archive for the robot concept;
- board layer exports used as visual and fabrication references.

These files are the starting point for the physical design. They should help contributors understand the robot shape, board placement, mechanical constraints, and the direction of the first prototype.

## Learning Philosophy

Hornsby should help people learn by doing.

Instead of teaching robotics as abstract theory, the robot should turn ideas into immediate experiments:

- write code;
- run it on the robot;
- watch what happens;
- measure the result;
- improve the behavior.

This makes the project useful for beginners, makers, students, and contributors who want to understand how software, electronics, mechanics, and AI come together.

## Project Values

- **Open learning:** the project should be understandable and welcoming to contributors.
- **Low cost:** decisions should favor parts and designs that can be reproduced affordably.
- **Real behavior:** examples should control actual hardware whenever possible.
- **Clear documentation:** every subsystem should be explained well enough for new contributors to continue the work.
- **Incremental progress:** start small, make it work, then improve it.

## Roadmap

Early milestones:

- document the first mechanical concept;
- define the electronics and control architecture;
- create the first motor-control test;
- add a simple programming exercise;
- add camera or sensor input;
- create a basic computer vision demo;
- build toward autonomous behavior.

## Collaboration

Hornsby is open from the beginning. Contributions are welcome in hardware, firmware, software, documentation, mechanical design, testing, education, and creative experiments.

The best contributions are practical, reproducible, and easy for the next person to understand.
