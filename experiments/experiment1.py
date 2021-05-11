import bodyposition as bp
import bodyposition.compare





for organ_label in ["lungs", "sagittal"]:
    for dataset in ["3DIrcadb1", "sliver07"]:
        bp.compare.compare(sdf_type=organ_label, dataset=dataset)