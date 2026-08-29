# illustrator/

Vector figures for the paper and the SSDM deck, drawn in Adobe Illustrator
and exported from it. Each figure keeps three files plus the script that
generated it, so a figure can be regenerated instead of re-drawn by hand.

| file | use |
|---|---|
| `*.ai` | editable source (saved Illustrator CC 2013+ compatible) |
| `*.png` | 600 dpi raster — slides |
| `*.svg` | vector, fonts outlined — manuscript |
| `*.jsx` | the ExtendScript that draws it |

## constant_capacitance_model

The DQD device model: V_g1 and V_g2 couple to their own dots through
C_g1d1 / C_g2d2 and to the opposite dot through the cross-capacitances
C_g1d2 / C_g2d1; QD1 and QD2 are coupled by the interdot capacitance C_m;
the sensor branch runs QD2 - C_s1d2 - QDs - C_s1g3 - V_gs.

PNG is 3583 x 4917 px (430 x 590 pt at 600 dpi).

## Regenerating

Illustrator must be running. From PowerShell:

```powershell
$ai = New-Object -ComObject Illustrator.Application
$ai.DoJavaScriptFile("F:\Seminar Fujita\projects\static_RBC_test_noise_8_15_ML_edited\illustrator\constant_capacitance_model.jsx")
```

The script opens a NEW document, so anything already open is untouched.
Do not call `$ai.Quit()` — it closes documents with unsaved changes.

Geometry lives at the top of the `.jsx` (`xL`, `xR`, `yGate`, `yDot`,
`ySens`, `yVgs`, radii); labels and export settings are at the bottom.
Coordinates in the script are y-down; `Y()` flips them to Illustrator's
convention.
