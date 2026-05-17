from depth_anything_3.specs import Prediction


def export(
    prediction: Prediction,
    export_format: str,
    export_dir: str,
    **kwargs,
):
    if "-" in export_format:
        for single_format in export_format.split("-"):
            export(prediction, single_format, export_dir, **kwargs)
        return

    if export_format == "glb":
        from .glb import export_to_glb

        export_to_glb(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format == "mini_npz":
        from .npz import export_to_mini_npz

        export_to_mini_npz(prediction, export_dir)
    elif export_format == "npz":
        from .npz import export_to_npz

        export_to_npz(prediction, export_dir)
    elif export_format == "feat_vis":
        from .feat_vis import export_to_feat_vis

        export_to_feat_vis(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format == "depth_vis":
        from .depth_vis import export_to_depth_vis

        export_to_depth_vis(prediction, export_dir)
    elif export_format == "gs_ply":
        from .gs import export_to_gs_ply

        export_to_gs_ply(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format == "gs_video":
        from .gs import export_to_gs_video

        export_to_gs_video(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format == "colmap":
        from .colmap import export_to_colmap

        export_to_colmap(prediction, export_dir, **kwargs.get(export_format, {}))
    else:
        raise ValueError(f"Unsupported export format: {export_format}")


__all__ = [
    "export",
]
