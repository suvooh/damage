Step 8 - Glare / light filter
=============================

Lamps, LED strips and reflections are near-white blobs the model can misread as damage. A halo around clipped pixels (red in glare_zone.png) is computed; a damage region that lies >= 85% inside it is NOT trusted as damage and the photo is sent to manual review instead.
Regions kept as damage: 1; regions moved to review as glare: 0.
