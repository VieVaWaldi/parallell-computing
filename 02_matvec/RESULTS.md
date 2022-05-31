For i=1
GPU timing in ms: h->d: 0.523776 kernelAx: 0.378880 kernelATx: 0.061280 d->h: 0.013120
CPU timing in ms: kernel: Ax: 6.842897  ATx: 7.119738

For i=10
GPU timing in ms: h->d: 41.112385 kernelAx: 4.824064 kernelATx: 1.140992 d->h: 0.050336
CPU timing in ms: kernel: Ax: 671.760578  ATx: 3303.941979

For i=30
GPU timing in ms: h->d: 364.049286 kernelAx: 38.586273 kernelATx: 10.802528 d->h: 0.109664
CPU timing in ms: kernel: Ax: 6044.114044  ATx: 34490.065208

=> GPU ist um Faktor 10 schneller auch mit Datenubertragung
=> ATx auf Host ist langsamer, aber auf GPU viel schneller
