from core.pipeline import LocalMedMoEPipeline


def test_smoke_infer():
    pipe = LocalMedMoEPipeline(
        ontology_path="config/ontology.json",
        entity_lexicon_path="data/lexicon/entities.json",
        docs_dir="data/docs",
        graph_path="data/graph/graph.json",
    )
    res = pipe.infer("Using the modified Pedersen hypothesis, explain how maternal hyperglycemia leads to fetal macrosomia.")
    assert res.answer
    assert res.qtype in {"fetal_growth_mechanism", "unknown"}
