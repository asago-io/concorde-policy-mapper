from concorde_policy_mapper.extract.models import RetrievalConfig
from concorde_policy_mapper.extract.parse import Chunk
from concorde_policy_mapper.extract.querygen import ChunkGroup, group_chunks


def test_retrieval_config_query_gen_default_false():
    rc = RetrievalConfig()
    assert rc.query_gen is False


def test_retrieval_config_query_gen_in_metadata():
    rc = RetrievalConfig(query_gen=True)
    meta = rc.to_metadata()
    assert meta["query_gen"] is True


def test_retrieval_config_query_gen_false_in_metadata():
    rc = RetrievalConfig(query_gen=False)
    meta = rc.to_metadata()
    assert meta["query_gen"] is False


def test_group_chunks_by_section():
    chunks = [
        Chunk(text="a", source="doc.pdf", index=0, section="Intro"),
        Chunk(text="b", source="doc.pdf", index=1, section="Intro"),
        Chunk(text="c", source="doc.pdf", index=2, section="Privacy"),
        Chunk(text="d", source="doc.pdf", index=3, section="Privacy"),
        Chunk(text="e", source="doc.pdf", index=4, section="Privacy"),
    ]
    groups = group_chunks(chunks)
    assert len(groups) == 2
    assert groups[0] == ChunkGroup(chunk_indices=[0, 1], section="Intro")
    assert groups[1] == ChunkGroup(chunk_indices=[2, 3, 4], section="Privacy")


def test_group_chunks_size_cap_splits():
    chunks = [
        Chunk(text=f"chunk {i}", source="doc.pdf", index=i, section="Big")
        for i in range(8)
    ]
    groups = group_chunks(chunks, max_group_size=3)
    assert len(groups) == 3
    assert groups[0].chunk_indices == [0, 1, 2]
    assert groups[1].chunk_indices == [3, 4, 5]
    assert groups[2].chunk_indices == [6, 7]
    assert all(g.section == "Big" for g in groups)


def test_group_chunks_none_merges_into_previous():
    chunks = [
        Chunk(text="a", source="doc.pdf", index=0, section="Intro"),
        Chunk(text="b", source="doc.pdf", index=1, section=None),
        Chunk(text="c", source="doc.pdf", index=2, section=None),
    ]
    groups = group_chunks(chunks)
    assert len(groups) == 1
    assert groups[0].chunk_indices == [0, 1, 2]
    assert groups[0].section == "Intro"


def test_group_chunks_none_at_start():
    chunks = [
        Chunk(text="a", source="doc.pdf", index=0, section=None),
        Chunk(text="b", source="doc.pdf", index=1, section=None),
        Chunk(text="c", source="doc.pdf", index=2, section="Privacy"),
    ]
    groups = group_chunks(chunks)
    assert len(groups) == 2
    assert groups[0].chunk_indices == [0, 1]
    assert groups[0].section is None
    assert groups[1].chunk_indices == [2]


def test_group_chunks_all_none_degrades_to_windows():
    chunks = [
        Chunk(text=f"chunk {i}", source="doc.pdf", index=i, section=None)
        for i in range(7)
    ]
    groups = group_chunks(chunks, max_group_size=3)
    assert len(groups) == 3
    assert groups[0].chunk_indices == [0, 1, 2]
    assert groups[1].chunk_indices == [3, 4, 5]
    assert groups[2].chunk_indices == [6]


def test_group_chunks_non_overlapping():
    chunks = [
        Chunk(text="a", source="doc.pdf", index=0, section="A"),
        Chunk(text="b", source="doc.pdf", index=1, section=None),
        Chunk(text="c", source="doc.pdf", index=2, section="B"),
        Chunk(text="d", source="doc.pdf", index=3, section="B"),
        Chunk(text="e", source="doc.pdf", index=4, section=None),
    ]
    groups = group_chunks(chunks)
    all_indices = []
    for g in groups:
        all_indices.extend(g.chunk_indices)
    assert sorted(all_indices) == [0, 1, 2, 3, 4]
    assert len(all_indices) == len(set(all_indices))


def test_group_chunks_single_chunk():
    chunks = [Chunk(text="only", source="doc.pdf", index=0, section="Solo")]
    groups = group_chunks(chunks)
    assert len(groups) == 1
    assert groups[0].chunk_indices == [0]


def test_group_chunks_empty():
    groups = group_chunks([])
    assert groups == []


def test_group_chunks_none_merges_respects_size_cap():
    chunks = [
        Chunk(text=f"chunk {i}", source="doc.pdf", index=i, section="A")
        for i in range(4)
    ] + [
        Chunk(text="null chunk", source="doc.pdf", index=4, section=None),
    ]
    groups = group_chunks(chunks, max_group_size=3)
    # Section "A" splits into [0,1,2] and [3]. None merges into [3] -> [3,4]
    assert groups[0].chunk_indices == [0, 1, 2]
    assert groups[1].chunk_indices == [3, 4]
