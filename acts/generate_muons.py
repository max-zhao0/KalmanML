#!/usr/bin/env python3

import acts, acts.examples, os
from acts.examples.simulation import addParticleGun, MomentumConfig, EtaConfig, ParticleConfig, addFatras, addDigitization, ParticleSelectorConfig
from acts.examples.reconstruction import addSeeding, addCKFTracks, TrackSelectorConfig, TruthSeedRanges, CkfConfig
from acts.examples.odd import getOpenDataDetector
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--acts_dir', type=str, help = 'Path containing build of ACTS')
parser.add_argument('--out_dir', type=str, help = 'Path to store generated data')
parser.add_argument('--n_events', type=int, help = 'Number of events to generate')
parser.add_argument('--n_muons', type=int, default=6000, help = 'Number of muons per event')
parser.add_argument('--detector_type', type=str, default="generic", choices=['generic', 'odd'], help = 'ACTS detector model to use (odd or generic)')
parser.add_argument('--rnd_seed', type=int, default=1234, help = 'Random seed for data generation')
parser.add_argument('--chi2_cut', type=float default=15 help = 'Chi2 cut for track fitting')
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
    MaterialMap = geoDir+"data/odd-material-maps.root"
    DigiConfig = geoDir+"config/odd-digi-geometric-config.json"
    SeedingSel = geoDir+"config/odd-seeding-config.json"
    MaterialDeco = acts.IMaterialDecorator.fromFile(MaterialMap)
    detector, trackingGeometry, decorators = getOpenDataDetector(geoDir, mdecorator=MaterialDeco)
else:
    print("Choose odd or generic detector")

field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 2.0 * u.T))
rnd = acts.examples.RandomNumbers(seed=42)

s = acts.examples.Sequencer(events=args.n_events, numThreads=1, outputDir=str(outputDir))

addParticleGun(
    s,
    MomentumConfig(1.0 * u.GeV, 10.0 * u.GeV, transverse=True),
    EtaConfig(-3.0, 3.0, uniform=True),
    ParticleConfig(args.n_muons, acts.PdgParticle.eMuon, randomizeCharge=True),
    rnd=rnd,
    outputDirRoot=outputDir,
)

addFatras(
    s,
    trackingGeometry,
    field,
    ParticleSelectorConfig(eta=(-3.0, 3.0), pt=(150 * u.MeV, None), removeNeutral=True),
    rnd=rnd,
    outputDirRoot=outputDir,
)

addDigitization(
    s,
    trackingGeometry,
    field,
    digiConfigFile=DigiConfig,
    rnd=rnd,
    outputDirRoot=outputDir,
)

addSeeding(
    s,
    trackingGeometry,
    field,
    TruthSeedRanges(pt=(1.0 * u.GeV, None), eta=(-3.0, 3.0), nHits=(9, None)),
    geoSelectionConfigFile=SeedingSel,
    outputDirRoot=outputDir,
)

addCKFTracks(
    s,
    trackingGeometry,
    field,
    TrackSelectorConfig(
        pt=(1.0 * u.GeV, None),
        absEta=(None, 3.0),
        loc0=(-4.0 * u.mm, 4.0 * u.mm),
        nMeasurementsMin=6,
    ),
    CkfConfig(
        chi2CutOff=args.chi2cut
    ),
    outputDirRoot=outputDir,
    writeCovMat=True
)

s.run()
