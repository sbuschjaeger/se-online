You can generate these files via the Moa GUI. Download MOA (https://moa.cms.waikato.ac.nz/), start it and copy/paste it into "Other Tasks" -> Right Click on the "configure" text-field. Paste & Run  

```
WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.LEDGeneratorDrift -d 1)   -d (ConceptDriftStream -s (generators.LEDGeneratorDrift -d 3) -d (ConceptDriftStream -s (generators.LEDGeneratorDrift -d 5)  -d (generators.LEDGeneratorDrift -d 7) -w 50 -p 250000 ) -w 50 -p 250000 ) -w 50 -p 250000) -m 1000000 -f /home/sbuschjaeger/projects/psgd-ensemble/online/led_a.arff
```

```
WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.LEDGeneratorDrift -d 1)   -d (ConceptDriftStream -s (generators.LEDGeneratorDrift -d 3) -d (ConceptDriftStream -s (generators.LEDGeneratorDrift -d 5)  -d (generators.LEDGeneratorDrift -d 7) -w 50000 -p 250000 ) -w 50000 -p 250000 ) -w 50000 -p 250000) -m 1000000 -f /home/sbuschjaeger/projects/psgd-ensemble/online/led_g.arff
```

```
WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.AgrawalGenerator -f 1) -d (ConceptDriftStream -s (generators.AgrawalGenerator -f 2) -d (ConceptDriftStream -s (generators.AgrawalGenerator )   -d (generators.AgrawalGenerator -f 4) -w 50 -p 250000 ) -w 50 -p 250000 ) -w 50 -p 250000) -m 1000000 -f /home/sbuschjaeger/projects/psgd-ensemble/online/agrawal_a.arff
```

```
WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.AgrawalGenerator -f 1) -d (ConceptDriftStream -s (generators.AgrawalGenerator -f 2) -d (ConceptDriftStream -s (generators.AgrawalGenerator )   -d (generators.AgrawalGenerator -f 4) -w 50000 -p 250000 ) -w 50000 -p 250000 ) -w 50000 -p 250000) -m 1000000 -f /home/sbuschjaeger/projects/psgd-ensemble/online/agrawal_g.arff
```

```
WriteStreamToARFFFile -s (generators.RandomRBFGeneratorDrift -c 5 -s .0001) -m 1000000 -f /home/sbuschjaeger/projects/psgd-ensemble/online/rbf_m.arff
WriteStreamToARFFFile -s (generators.RandomRBFGeneratorDrift -c 5 -s .001) -m 1000000 -f /home/sbuschjaeger/projects/psgd-ensemble/online/rbf_f.arff
```
