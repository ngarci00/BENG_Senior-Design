# Video Dataset Notes

This file documents the raw ETI video naming scheme, required annotation
metadata, label conventions, and the current video inventory.

## Video Naming Convention

Videos are named as:

`<number>.<angle><outcome>.avi`

Examples:

- `1.RG.avi`
- `24.RB.avi`
- `41.MG.avi`

Angle and outcome codes:

- `RG`: right-side good
- `RB`: right-side bad
- `LG`: left-side good
- `LB`: left-side bad
- `MG`: middle-side good
- `MB`: middle-side bad

## Required Timing Metadata

Please add these timestamps when reviewing and annotating videos:

- `t_attempt_start`
- `t_vocal_cords_visible`
- `t_tube_pass`
- `t_attempt_end`

Notes:

- `t_vocal_cords_visible` means the vocal cords are fully visible.
- `t_tube_pass` means the tube passes between the black guide lines as taught.

## Label Names

Keep these class names consistent across all annotations:

- `epiglottis`
- `endotracheal_tube`
- `vocal_cords`
- `arytenoids`
- `esophagus`

## Convert Videos To Frames

If you need to convert a video into frames for annotation, use:

```bash
.venv/bin/video-toimg nameofvideo.avi
```

Reference:

- https://github.com/wkentaro/labelme/blob/main/examples/video_annotation/README.md

If the command does not work, activate the repo virtualenv first:

```bash
source .venv/bin/activate
.venv/bin/video-toimg nameofvideo.avi
```

## Open Videos Locally

To open a video from the terminal with IINA:

```bash
open -a IINA name_file.avi
```

## Turn LabelMe JSON Back Into A Video

Use the repo-local tools rather than the old external helpers.

1. Normalize rectangle JSON files:

```bash
python3 normrectangles.py "input_folder"
```

2. Export LabelMe visualizations:

```bash
.venv/bin/python scripts/export_labelme_visualization.py "input_folder"
```

This creates a folder with a `.export` suffix.

3. Build a video from the exported visualization frames:

```bash
.venv/bin/python scripts/video_from_images.py output_file.mp4 -i "name_file.export/*/visualization.jpg" --fps 30
```

## Video Inventory

This is the working human-readable inventory. Update it as new videos are added.

### Taken 1/21

1. Right side good.
2. Middle good.
3. Left good.
4. Right bad.
   Reason: not far enough.
5. Middle good.
   Note: hit vocals.
6. Right good.
7. Bad glidescope insert. Tube inserted into esophagus.
8. Middle bad. Tube in the esophagus.
9. Identifying epiglottis. Contains perfect view.
   Status: omit.
10. Middle bad. Hitting epiglottis.
11. Middle good.
12. Middle bad. Glidescope too deep and inserted into stomach.
13. Middle bad. Multiple attempts and inserted into stomach.
   Note: middle angle makes you miss seeing epiglottis.
14. Right good. Tongue is less in the way.
15. Left bad. Bad visualization.
16. Right bad. Difficult insertion.
17. Right bad. Difficult insertion.
18. Right good.
19. Left good.
   Note: metal stylet should not be visible past the tube.
20. Middle good.
   Note: hit the wall between esophagus and vocal cords. Blade angled left to make tube insertion easier.
21. Right good. Faster. Tube can be inserted straight instead of at an angle.
22. Right good with educational explanations during the video.
   Status: omit.
23. Right good. Long video.

### Taken 2/6

24. Right bad. Fail.
25. Right bad. Did not see vocal cords.
26. Right bad. Fail.
   Note: rectangle videos labeled so far.
27. Right bad.
28. Center bad. No view of vocal cords. Blind insertion.
29. Center bad.
30. Center bad. Too deep.
31. Moving around with no clear angle. Bad.
32. Upside down. Bad.
33. Right bad. Tube never passed through cords. Trauma inducing.
34. Middle bad.
35. Right bad.
36. Left bad.
37. Bad.
38. Center bad. Difficult to find view.
39. Left bad. Bad view.
40. Right good.
41. Middle good.
42. Right good. Seven seconds.
43. Right good.
44. Right good.
45. Middle good. Some talking during video.
46. Right good.
47. Middle good.
48. Right/middle good.
49. Right/middle good.
50. Right good.
51. Right good.
52. Middle good. Slow.

### Extra

53. Right bad. No visualization of vocal cords and lots of moving.
54. Middle good.

## Storage Notes

- Keep `raw_videos` and any generated frames local to your machine.
- Source videos should live in the shared Google Drive folder.
