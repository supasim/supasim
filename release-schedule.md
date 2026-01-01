# Release schedule

Currently, no regular release schedule is planned. Instead, I will be using the following roadmap.

## Version 0.1

This will be the first minor version to be released onto crates.io; previous releases were runnable but mainly created to reserve the names.

This is planned to release immediately after #93, representing a point where much of the internals is functional and extensible. This will remove unnecessary parts of the API, like buffers only living in one place, and leave an API that will look similar to what will happen in the future. It will be usable for real projects, and should be simple to use.

## Version 0.2

This is the cleanup version for 0.1. You can follow its progress on [the GitHub milestone](https://github.com/supasim/supasim/milestone/4). This will likely be the first release that will be advertised publicly. At this point, it should be mostly bug-free and usable, with a near-final API.

## Version 0.3 and later

For version 0.3 and later, we will likely focus on adding more optional features or improving performance. At this point we might switch to more free-flowing or scheduled releases.

## Version 1.0

Unlike other GPU APIs like wgpu, I plan to rarely change the API, partly due to the narrow focus on compute. New features will be few and far between, and rarely require breaking changes. For this reason, it is a goal to eventually reach 1.0.

There is [a GitHub milestone for 1.0](https://github.com/supasim/supasim/milestone/3), representing some features I will be considering necessary, but this list isn't exhaustive.
