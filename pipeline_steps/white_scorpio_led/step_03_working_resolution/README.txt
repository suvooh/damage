Step 3 - Resize into the model's working range
==============================================

The model was trained on photos with a 1,024-1,600 px long side. This photo (501 px long side) is resized to 816 x 1024 px (bicubic) so thin scratches keep enough pixels.
All probability maps are mapped back to the original size at the end.
