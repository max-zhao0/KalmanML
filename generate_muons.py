#!/usr/bin/env python3

import pathlib, acts, acts.examples, os
from acts.examples.simulation import addParticleGun, MomentumConfig, EtaConfig, ParticleConfig, addFatras, addDigitization, ParticleSelectorConfig
from acts.examples.reconstruction import addSeeding, addCKFTracks, CKFPerformanceConfig, TrackSelectorRanges, TruthSeedRanges
from acts.examples.odd import getOpenDataDetector

actsdir = "/global/homes/j/jmw464/ATLAS/Software/acts21.2/"
datadir = "/global/cfs/cdirs/atlas/jmw464/mlkf_data/python"
nevents = 200
nmuons = 6000 #muons per event
detector = "generic"

u = acts.UnitConstants
outputDir = datadir+"ttbar"+str(npileup)+"_"+str(nevents)+"/"
#outputDirCsv = outputDir+"csv/"

if not os.path.exists(outputDir): os.mkdir(outputDir)
#if not os.path.exists(outputDirCsv): os.mkdir(outputDirCsv)

if detector == "generic":
    DigiConfig = actsdir+"/Examples/Algorithms/Digitization/share/default-geometric-config-generic.json"
    SeedingSel = actsdir+"/Examples/Algorithms/TrackFinding/share/geoSelection-genericDetector.json"
    detector, trackingGeometry, decorators = acts.examples.GenericDetector.create()
elif detector == "odd":
    geoDir = actsdir+"thirdparty/OpenDataDetector/"
    MaterialMap = geoDir+"data/odd-material-maps.root"
    DigiConfig = geoDir+"config/odd-digi-geometric-config.json"
    SeedingSel = geoDir+"config/odd-seeding-config.json"
    MaterialDeco = acts.IMaterialDecorator.fromFile(MaterialMap)
    detector, trackingGeometry, decorators = getOpenDataDetector(geoDir, mdecorator=MaterialDeco)
else:
    print("Choose odd or generic detector")

field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 2.0 * u.T))
rnd = acts.examples.RandomNumbers(seed=42)

s = acts.examples.Sequencer(events=nevents, numThreads=1, outputDir=str(outputDir))

addParticleGun(
    s,
    MomentumConfig(1.0 * u.GeV, 10.0 * u.GeV, transverse=True),
    EtaConfig(-3.0, 3.0, uniform=True),
    ParticleConfig(nmuons, acts.PdgParticle.eMuon, randomizeCharge=True),
    rnd=rnd,
    outputDirRoot=outputDir,
#    outputDirCsv=outputDirCsv,
)

addFatras(
    s,
    trackingGeometry,
    field,
    ParticleSelectorConfig(eta=(-3.0, 3.0), pt=(150 * u.MeV, None), removeNeutral=True),
    rnd=rnd,
    outputDirRoot=outputDir,
#    outputDirCsv=outputDirCsv,
)

addDigitization(
    s,
    trackingGeometry,
    field,
    digiConfigFile=DigiConfig,
    rnd=rnd,
    outputDirRoot=outputDir,
#    outputDirCsv=outputDirCsv,
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
    CKFPerformanceConfig(ptMin=1.0 * u.GeV, nMeasurementsMin=6),
    TrackSelectorRanges(pt=(1.0 * u.GeV, None), absEta=(None, 3.0), removeNeutral=True),
    outputDirRoot=outputDir,
#    outputDirCsv=outputDirCsv,
)

s.run()
