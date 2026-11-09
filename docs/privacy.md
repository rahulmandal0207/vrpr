# Privacy

This is engineering guidance, not legal advice. Check your obligations before processing real
traffic. In India, start with the Digital Personal Data Protection Act, 2023.

## What VRPR stores

A registration number can identify a vehicle's owner, and a crop or video can also show people,
places and movements over time.

| Data | Location | Protection |
| --- | --- | --- |
| Plate numbers | `plates.csv`, `plates.jsonl` | None: these are in full |
| Plate images | `crops/` | None. Turn off with `--no-crops` |
| Full video | `annotated.mp4` | Off by default, on with `--save-video` |
| Logs | stderr | Plate numbers are masked, for example `HR26****37` |

Git ignores `output/`, `data/` and video files, so results stay out of the repository. The one
exception is the public-domain sample clip in `samples/`.

## Public demo

In public mode (`streamlit_app.py`), anyone with the link can upload a video:

- Each browser session gets its own temporary folder. Other visitors cannot see its uploads or
  results, and folders older than about 2 hours are deleted.
- Results are not stored anywhere else, and no plate images or annotated videos are kept after
  the session folder is deleted.
- The page tells visitors that plates are personal data and to upload only footage they may
  share.
- The host, Streamlit Community Cloud, keeps its own server logs. Plate numbers in VRPR's logs
  are masked.

If you publish the demo, you are responsible for it. Take it down if it is used to track people.

## Practices to follow

- Have a lawful reason, and tell people when required, for example with signage.
- Keep only what the purpose needs. Use `--no-crops` if you do not need the images.
- Restrict access to output folders, and encrypt the disk they are stored on.
- Delete results on a schedule. VRPR has no retention job, so this is up to the operator.
- Treat a reported plate as a lead, not proof. OCR makes mistakes, so have a person check before
  anything affects someone.

## Vehicle owner lookups

Earlier versions included a disabled call to a third-party API that returned owner details, and it
sent a hardcoded statement of consent. It was removed on purpose: the people whose plates are
scanned have not consented to that lookup, and software cannot give consent on their behalf. Do not
add a lookup like this without a lawful basis of your own.
