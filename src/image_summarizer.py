"""
Image summaries for the Documents figures and tables
"""


IMAGE_SUMMARIES = { # Predefined image summaries & Very detailed descriptions using VLMs
    "extracted_image_1.png": "Figure Summary: Cette figure illustre comment un défaut valvulaire, en présence de bactéries dans le sang, mène à la formation de végétations qui détruisent la valve cardiaque et provoquent une insuffisance cardiaque.",
    
    "extracted_image_2.png": "Figure Summary: Ce schéma détaille les agents microbiologiques responsables des endocardites infectieuses, en soulignant la nette prédominance des Streptocoques et des Staphylocoques.",
    
}


def get_image_summaries():
    """Return the predefined image summaries"""
    return IMAGE_SUMMARIES


def add_image_summary(image_name: str, summary: str):
    """
    Add a new image summary to the collection
    
    Args:
        image_name: Name/identifier for the image
        summary: Text summary of the image
    """
    IMAGE_SUMMARIES[image_name] = summary


def load_summaries_from_file(filepath: str):
    """
    Load image summaries from a JSON file
    
    Args:
        filepath: Path to JSON file with format {"image_name": "summary"}
    """
    import json
    with open(filepath, 'r') as f:
        summaries = json.load(f)
    IMAGE_SUMMARIES.update(summaries)
