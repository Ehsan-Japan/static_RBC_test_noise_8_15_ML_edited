/*  Constant-capacitance model of the DQD device + charge sensor.
 *  Draws the circuit as vector art in a NEW document, then saves .ai
 *  and exports PNG (600 dpi) and SVG into the project folder.
 *  Coordinates below are "y increases downward"; Y() flips them.
 */
#target illustrator

var OUT = "F:\\Seminar Fujita\\projects\\static_RBC_test_noise_8_15_ML_edited\\illustrator\\";
var W = 430, H = 590;

app.coordinateSystem = CoordinateSystem.ARTBOARDCOORDINATESYSTEM;
var doc = app.documents.add(DocumentColorSpace.RGB, W, H);

// ── colours ───────────────────────────────────────────────────────────────
function rgb(r, g, b) {
    var c = new RGBColor(); c.red = r; c.green = g; c.blue = b; return c;
}
var INK = rgb(0, 0, 0);
var WHITE = rgb(255, 255, 255);

var LW = 1.4;          // wire weight
var PW = 2.0;          // capacitor plate weight

function Y(v) { return -v; }

// ── font ──────────────────────────────────────────────────────────────────
var FONT = null;
var names = ["ArialMT", "Arial", "Helvetica", "HelveticaNeue"];
for (var i = 0; i < names.length; i++) {
    try { FONT = app.textFonts.getByName(names[i]); break; } catch (e) {}
}

// ── primitives ────────────────────────────────────────────────────────────
function line(x1, y1, x2, y2, w) {
    var p = doc.pathItems.add();
    p.setEntirePath([[x1, Y(y1)], [x2, Y(y2)]]);
    p.filled = false;
    p.stroked = true;
    p.strokeColor = INK;
    p.strokeWidth = (w === undefined) ? LW : w;
    return p;
}

function circle(cx, cy, r, fill) {
    var p = doc.pathItems.ellipse(Y(cy - r), cx - r, 2 * r, 2 * r);
    p.stroked = true;
    p.strokeColor = INK;
    p.strokeWidth = LW;
    p.filled = true;
    p.fillColor = (fill === undefined) ? WHITE : fill;
    return p;
}

/* A capacitor drawn ON the segment (x1,y1)-(x2,y2): wire, two plates
   perpendicular to the wire, wire.  half = plate half-length, gap = plate
   separation. */
function cap(x1, y1, x2, y2, half, gap) {
    half = (half === undefined) ? 13 : half;
    gap = (gap === undefined) ? 7 : gap;
    var dx = x2 - x1, dy = y2 - y1;
    var L = Math.sqrt(dx * dx + dy * dy);
    var ux = dx / L, uy = dy / L;         // along the wire
    var px = -uy, py = ux;                // perpendicular
    var mx = (x1 + x2) / 2, my = (y1 + y2) / 2;

    var a = [mx - ux * gap / 2, my - uy * gap / 2];   // near plate centre
    var b = [mx + ux * gap / 2, my + uy * gap / 2];   // far plate centre

    line(x1, y1, a[0], a[1]);                          // lead in
    line(b[0], b[1], x2, y2);                          // lead out
    line(a[0] - px * half, a[1] - py * half,
         a[0] + px * half, a[1] + py * half, PW);      // plate 1
    line(b[0] - px * half, b[1] - py * half,
         b[0] + px * half, b[1] + py * half, PW);      // plate 2
}

/* Text with a trailing subscript.  main = upright part, sub = subscript. */
function label(x, y, main, sub, size, anchor) {
    size = (size === undefined) ? 14 : size;
    var t = doc.textFrames.add();
    t.contents = main + (sub === undefined ? "" : sub);
    var ca = t.textRange.characterAttributes;
    if (FONT) ca.textFont = FONT;
    ca.size = size;
    ca.fillColor = INK;
    if (sub !== undefined && sub !== "") {
        for (var i = main.length; i < t.contents.length; i++) {
            var c = t.characters[i].characterAttributes;
            c.size = size * 0.66;
            c.baselineShift = -size * 0.22;
        }
    }
    // anchor: "l" left (default), "c" centred on x, "r" right-aligned to x
    if (anchor === "c") t.paragraphs[0].paragraphAttributes.justification =
        Justification.CENTER;
    var wdt = t.width;
    if (anchor === "c") t.left = x - wdt / 2;
    else if (anchor === "r") t.left = x - wdt;
    else t.left = x;
    t.top = Y(y);
    return t;
}

// ── geometry (y down) ─────────────────────────────────────────────────────
var xL = 78, xR = 338;      // the two gate / dot columns
var yGate = 42;             // Vg1, Vg2
var yDot = 292;             // QD1, QD2
var ySens = 424;            // QDs
var yVgs = 534;             // Vgs
var rTerm = 11, rDot = 33, rSens = 26;

// vertical gate capacitors:  Vg1 - Cg1d1 - QD1,  Vg2 - Cg2d2 - QD2
cap(xL, yGate + rTerm, xL, yDot - rDot, 15, 8);
cap(xR, yGate + rTerm, xR, yDot - rDot, 15, 8);

// crossed cross-capacitances: Vg1 -> QD2 and Vg2 -> QD1.
// Capacitor symbol sits high on each diagonal, as in the reference figure.
function diagonal(xa, ya, xb, yb, tFrac) {
    var dx = xb - xa, dy = yb - ya;
    var L = Math.sqrt(dx * dx + dy * dy);
    var ux = dx / L, uy = dy / L;
    var cx = xa + ux * L * tFrac, cy = ya + uy * L * tFrac;
    cap(xa, ya, cx, cy, 13, 7);          // lead + plates near the gate
    line(cx, cy, xb, yb);                 // the long run down to the dot
}
diagonal(xL + rTerm * 0.75, yGate + rTerm * 0.75,
         xR - rDot * 0.72, yDot - rDot * 0.72, 0.30);
diagonal(xR - rTerm * 0.75, yGate + rTerm * 0.75,
         xL + rDot * 0.72, yDot - rDot * 0.72, 0.30);

// interdot capacitance Cm
cap(xL + rDot, yDot, xR - rDot, yDot, 15, 8);

// sensor branch: QD2 - Cs1d2 - QDs - Cs1g3 - Vgs
cap(xR, yDot + rDot, xR, ySens - rSens, 14, 7);
cap(xR, ySens + rSens, xR, yVgs - rTerm, 14, 7);

// ── nodes ─────────────────────────────────────────────────────────────────
circle(xL, yGate, rTerm);
circle(xR, yGate, rTerm);
circle(xL, yDot, rDot);
circle(xR, yDot, rDot);
circle(xR, ySens, rSens);
circle(xR, yVgs, rTerm);

// ── labels ────────────────────────────────────────────────────────────────
label(xL - rTerm - 8, yGate - 8, "V", "g1", 15, "r");
label(xR + rTerm + 8, yGate - 8, "V", "g2", 15, "l");
label(xR + rTerm + 8, yVgs - 8, "V", "gs", 15, "l");

label(xL, yDot + 6, "QD", "1", 13, "c");
label(xR, yDot + 6, "QD", "2", 13, "c");
label(xR, ySens + 5, "QD", "s", 11.5, "c");

label(xL - 22, (yGate + yDot) / 2 - 10, "C", "g1d1", 14, "r");
label(xR + 22, (yGate + yDot) / 2 - 10, "C", "g2d2", 14, "l");

label(xL + 62, yGate + 18, "C", "g1d2", 14, "l");
label(xR - 62, yGate + 18, "C", "g2d1", 14, "r");

label((xL + xR) / 2, yDot + 22, "C", "m", 14, "c");

label(xR + 22, (yDot + ySens) / 2 - 10, "C", "s1d2", 14, "l");
label(xR + 22, (ySens + yVgs) / 2 - 10, "C", "s1g3", 14, "l");

// ── save and export ───────────────────────────────────────────────────────
doc.selection = null;

// .ai source
var aiOpts = new IllustratorSaveOptions();
aiOpts.compatibility = Compatibility.ILLUSTRATOR17;
doc.saveAs(new File(OUT + "constant_capacitance_model.ai"), aiOpts);

// PNG at 600 dpi
var png = new ExportOptionsPNG24();
png.antiAliasing = true;
png.transparency = false;
png.artBoardClipping = true;
png.horizontalScale = 600 / 72 * 100;
png.verticalScale = 600 / 72 * 100;
doc.exportFile(new File(OUT + "constant_capacitance_model.png"),
               ExportType.PNG24, png);

// SVG for the manuscript
var svg = new ExportOptionsSVG();
svg.embedRasterImages = false;
svg.fontType = SVGFontType.OUTLINEFONT;
doc.exportFile(new File(OUT + "constant_capacitance_model.svg"),
               ExportType.SVG, svg);

"drawn and exported to " + OUT;
