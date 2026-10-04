import pytest
from conftest import ROOT
from integrations.teacher.simulator import TeacherSimulator
from integrations.teacher.provider import HelpRequest
from training.learning_package.schema import validate, InvalidLearningPackage, LearningPackage
import dataclasses


def lp():
    return TeacherSimulator().respond(HelpRequest("compare numbers relation", 2))


def test_valid_package_passes():
    validate(lp())


def test_memory_or_knowledge_never_trains_neural_modules():
    for kind in ("MEMORY", "KNOWLEDGE"):
        with pytest.raises(InvalidLearningPackage):
            validate(dataclasses.replace(lp(), info_kind=kind))


def test_validation_leak_rejected():
    p = lp(); p.validation_x[0] = list(p.train_x[0]); p.validation_y[0] = p.train_y[0]
    with pytest.raises(InvalidLearningPackage):
        validate(p)


def test_missing_provenance_and_bad_labels_rejected():
    p = lp(); p.provenance = {}
    with pytest.raises(InvalidLearningPackage):
        validate(p)
    p = lp(); p.train_y[0] = 99
    with pytest.raises(InvalidLearningPackage):
        validate(p)


def test_unknown_dependency_rejected():
    p = lp(); p.existing_dependencies = ["nope"]
    with pytest.raises(InvalidLearningPackage):
        validate(p, known_capabilities=set())
