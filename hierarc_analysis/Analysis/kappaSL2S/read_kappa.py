import pickle
import toml
from pathlib import Path



def load_sl2s_lenses():
	path = Path(__file__).parents[0]
	return load_kappa_folder(path)

def load_kappa_folder(path: Path):
	output = {}
	lens_list_path = path / "lenses.toml"
	known_lenses = None
	if lens_list_path.exists():
		with open(lens_list_path, "r") as f:
			known_lenses = toml.load(f)["known_lenses"]

	kappa_files = [f for f in path.glob("*.kappa")]
	if not kappa_files:
		raise FileNotFoundError(f"No kappa files found at path {path}")

	found_lenses = [f.stem for f in  kappa_files]
	if known_lenses:
		if (missing := set(known_lenses) - set(found_lenses)):
			print(f"Warning... Missing kappa files for lenses: {','.join(missing)}") 
		if (extras := set(found_lenses) - set(known_lenses)):
			print(f"Warning... found extra lenses: {','.join(extras)}.... Loading anyway...") 


	for file in kappa_files:
		with open(file, "rb") as f:
			data = pickle.load(f)
			output.update({file.stem: data})
	return output
