| condition | score | gain | interference | battery (all) | battery (visible) | battery (hidden) |
|---|---|---|---|---|---|---|
| No interface | — | — | — | 0.379 | 0.829 | 0.283 |
| Hand-designed, global | -0.183 | +0.661 | +0.834 | 0.740 | 0.785 | 0.730 |
| Hand-designed, local | +0.121 | +0.131 | +0.000 | 0.460 | 0.819 | 0.384 |
| Random search, adaptive | +0.378 | +0.390 | +0.002 | 0.514 | 0.796 | 0.454 |
| CCS, global plasticity | +0.513 | +0.531 | +0.008 | 0.636 | 0.792 | 0.603 |
| CCS, local plasticity (Rc only) | +0.184 | +0.194 | +0.000 | 0.403 | 0.825 | 0.314 |
| CCS, adaptive plasticity | +0.499 | +0.509 | +0.005 | 0.620 | 0.795 | 0.583 |

Selected environments:
- world 0 ccs-adaptive: score +0.538 — carries z[0, 1, 2, 4, 5, 6, 7] | linear code, noise 0.05 | objective rec=1.0, align=0.0 | input=channel | open Rc+R2 for 150 | lr=0.003
- world 0 ccs-global: score +0.543 — carries z[1, 2, 3, 4, 5, 6, 7] | linear code, noise 0.05 | objective rec=0.5, align=0.0 | input=channel | open R1+Rc+R2 for 600 | lr=0.003
- world 0 ccs-local: score +0.216 — carries z[1, 2, 3, 4, 5, 6, 7] | linear code, noise 0.05 | objective rec=1.0, align=0.0 | input=channel | open Rc for 600 | lr=0.003
- world 0 random-adaptive: score +0.418 — carries z[0, 1, 2, 4, 5, 6] | linear code, noise 0.05 | objective rec=1.0, align=0.0 | input=channel | open R1 for 300; open R2 for 300 | lr=0.003
- world 1 ccs-adaptive: score +0.521 — carries z[1, 2, 3, 4, 5, 6, 7] | linear code, noise 0.05 | objective rec=1.0, align=0.0 | input=channel | open Rc+R2 for 300; open R1+Rc+R2 for 150 | lr=0.003
- world 1 ccs-global: score +0.517 — carries z[0, 2, 3, 4, 5, 6, 7] | linear code, noise 0.05 | objective rec=1.0, align=0.0 | input=channel | open R1+Rc+R2 for 600 | lr=0.001
- world 1 ccs-local: score +0.147 — carries z[0, 2, 3, 4, 6, 7] | linear code, noise 0.05 | objective rec=1.0, align=0.0 | input=channel | open Rc for 600 | lr=0.003
- world 1 random-adaptive: score +0.409 — carries z[0, 1, 2, 3, 4, 5, 6] | linear code, noise 0.3 | objective rec=0.5, align=0.0 | input=channel | open R2 for 300 | lr=0.003
- world 2 ccs-adaptive: score +0.439 — carries z[1, 2, 3, 4, 5, 6] | linear code, noise 0.05 | objective rec=0.5, align=0.0 | input=channel | open R1+Rc+R2 for 150 | lr=0.003
- world 2 ccs-global: score +0.479 — carries z[0, 1, 3, 4, 5, 6, 7] | linear code, noise 0.3 | objective rec=1.0, align=0.0 | input=channel | open R1+Rc+R2 for 600 | lr=0.001
- world 2 ccs-local: score +0.189 — carries z[1, 2, 4, 5, 6, 7] | linear code, noise 0.3 | objective rec=1.0, align=1.0 | input=channel | open Rc for 600 | lr=0.003
- world 2 random-adaptive: score +0.308 — carries z[0, 1, 3, 4, 6] | linear code, noise 0.3 | objective rec=1.0, align=0.0 | input=channel | open R1+Rc for 300; open Rc+R2 for 600 | lr=0.003
