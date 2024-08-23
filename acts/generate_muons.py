#!/usr/bin/env python3

import acts, acts.examples, os
from acts.examples.simulation import addParticleGun, MomentumConfig, EtaConfig, ParticleConfig, PhiConfig, addFatras, addDigitization, ParticleSelectorConfig
from acts.examples.reconstruction import addSeeding, addCKFTracks, TrackSelectorConfig,TruthSeedRanges, CkfConfig #, addSeedFilterML, SeedFilterMLDBScanConfig
from acts.examples.odd import getOpenDataDetector
import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--acts_dir', type=str, help = 'Path containing build of ACTS')
parser.add_argument('--out_dir', type=str, help = 'Path to store generated data')
parser.add_argument('--n_events', type=int, help = 'Number of events to generate')
parser.add_argument('--n_muons', type=int, default=6000, help = 'Number of muons per event')
parser.add_argument('--detector_type', type=str, default="odd", choices=['generic', 'odd'], help = 'ACTS detector model to use (odd or generic)')
parser.add_argument('--rnd_seed', type=int, default=42, help = 'Random seed for data generation')
parser.add_argument('--chi2_cut', type=float, default=15, help = 'Chi2 cut for track fitting')
parser.add_argument('--pt_range', nargs=2, type=float, default=[1.0, 10.0], help = 'pT range for muons (GeV)')
parser.add_argument('--eta_range', nargs=2, type=float, default=[-3.0, 3.0], help = 'eta range for muons')
args = parser.parse_args()

u = acts.UnitConstants
outputDir = args.out_dir+"muon"+str(args.n_pileup)+"_"+str(args.n_events)+"/"
if not os.path.exists(outputDir): os.mkdir(outputDir)

if args.detector_type == "generic":
    DigiConfig = args.acts_dir+"/Examples/Algorithms/Digitization/share/default-geometric-config-generic.json"
    SeedingSel = args.acts_dir+"/Examples/Algorithms/TrackFinding/share/geoSelection-genericDetector.json"
    detector, trackingGeometry, decorators = acts.examples.GenericDetector.create()
elif args.detector_type == "odd":
    geoDir = args.acts_dir+"thirdparty/OpenDataDetector/"
    seedFilterModel = args.acts_dir+"/Examples/scripts/python/MLAmbiguityResolution/seedDuplicateClassifier.onnx"
    MaterialMap = geoDir+"data/odd-material-maps.root"
    DigiConfig = geoDir+"config/odd-digi-smearing-config.json"
    SeedingSel = geoDir+"config/odd-seeding-config.json"
    MaterialDeco = acts.IMaterialDecorator.fromFile(Path(MaterialMap))
    detector, trackingGeometry, decorators = getOpenDataDetector(odd_dir=Path(geoDir), mdecorator=MaterialDeco)
else:
    print("Choose odd or generic detector")

field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 2.0 * u.T))
rnd = acts.examples.RandomNumbers(seed=args.rnd_seed)

s = acts.examples.Sequencer(events=args.n_events, numThreads=-1, outputDir=str(outputDir))

addParticleGun(
    s,
    MomentumConfig(args.pt_range[0] * u.GeV, args.pt_range[1] * u.GeV, transverse=True),
    EtaConfig(args.eta_range[0], args.eta_range[1], uniform=True),
    PhiConfig(0.0, 360.0 * u.degree),
    ParticleConfig(args.n_muons, acts.PdgParticle.eMuon, randomizeCharge=True),
    rnd=rnd,
    outputDirRoot=outputDir,
)

addFatras(
    s,
    trackingGeometry,
    field,
    preSelectParticles=ParticleSelectorConfig(),
    rnd=rnd,
    enableInteractions=True,
    outputDirRoot=outputDir,
)

addDigitization(
    s,
    trackingGeometry,
    field,
    digiConfigFile=Path(DigiConfig),
    rnd=rnd,
    outputDirRoot=outputDir,
)

addSeeding(
    s,
    trackingGeometry,
    field,
    TruthSeedRanges(),
    initialSigmas=[1 * u.mm, 1 * u.mm, 1 * u.degree, 1 * u.degree, 0.1 * u.e / u.GeV, 1 * u.ns],
    initialSigmaPtRel=0.1,
    initialVarInflation=[1.0] * 6,
    geoSelectionConfigFile=Path(SeedingSel),
    outputDirRoot=outputDir,
)

#if args.detector_type == "odd":
#    addSeedFilterML(
#        s,
#        SeedFilterMLDBScanConfig(
#            epsilonDBScan=0.03, minPointsDBScan=2, minSeedScore=0.1
#        ),
#        onnxModelFile=Path(seedFilterModel),
#        outputDirRoot=outputDir,
#    )

addCKFTracks(
    s,
    trackingGeometry,
    field,
    TrackSelectorConfig(
        pt=(0.0 * u.GeV, None),
        absEta=(None, 3.0),
        loc0=(-4.0 * u.mm, 4.0 * u.mm),
        nMeasurementsMin=7,
        maxHoles=2,
        maxOutliers=2,
    ),
    CkfConfig(
        chi2CutOff=args.chi2cut,
        numMeasurementsCutOff=10,
        seedDeduplication=True,
        stayOnSeed=True,
        pixelVolumes={16, 17, 18},
        stripVolumes={23, 24, 25},
        maxPixelHoles=1,
        maxStripHoles=2,
    ),
    outputDirRoot=outputDir,
    writeCovMat=True
)

s.run()
