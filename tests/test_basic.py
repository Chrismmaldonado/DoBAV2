def test_import():
    """Test that the main modules can be imported."""
    try:
        import DobAEI
        import neural_cache_system
        import sqlite_nuclear_memory
        assert True
    except ImportError:
        assert False, "Failed to import main modules"
