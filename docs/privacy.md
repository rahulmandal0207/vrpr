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

Git ignores `output/`, `data/` and video files, so results stay out of the repository.

## Public demo

The public demo on the `demo` branch accepts uploads from anyone with the link. Its privacy
measures, such as private per-visitor sessions deleted after about 2 hours, are described in that
branch's copy of this file.

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

The plate details VRPR does show, the state, RTO office and area, and the year of a Bharat series
plate, come from the registration number and a public table of RTO codes bundled with VRPR. They
say where a number was issued, not who owns the vehicle, and no plate is sent anywhere to get
them.
