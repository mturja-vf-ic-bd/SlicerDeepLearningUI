from Asynchrony import Asynchrony
from geometry_image.tools.perform_sgim_sampling import *
from pathlib import Path


HOME = "/Users/mturja/PycharmProjects/geometry_image/"


def get_command(file, args):
    input_file, output_file = file
    command = args["PYTHON_PATH"] + " " + \
              os.path.join(HOME, "tools", "perform_sgim_sampling.py") + " " \
              + "-s {} -f {} -r {} -o {}".format(args["template"],
                                                 input_file,
                                                 args["resolution"],
                                                 output_file)
    return command


def run_geom_image(args):
    """
    Run this script to compute geometry image for a set of subjects in a directory
    """

    filenames = generate_file_names(args["sub_dir"], args["output"])
    total = len(filenames)
    c = 0
    for files in filenames:
        input_file, output_file = files
        args["feature_map"] = input_file
        args["output"] = output_file
        sgim_sampling_wrapper(args)
        c += 1
        val =(c + 1) * 100 // total
        if args["progressBar"] is not None:
            Asynchrony.RunOnMainThread(lambda: args["progressBar"].setValue(val))
    if args["progressBar"] is not None:
        Asynchrony.RunOnMainThread(lambda: args["progressBar"].setValue(val))


def generate_file_names(input_directory, output_directory):
    """
    This function generates input and output file names
    from the following directory structure.

    :param input_directory: Contains one folder per subject.
    Each of the folder will contain folders for each scalars.
    ============= Example Directory Structure =============
        <directory>
            <sub_id_1>
                <session_id>
                    - <modality 1>
                    - <modality 2>
                    - etc.
            <sub_id_2>
                <session_id>
                    - <modality 1>
                    - <modality 2>
                    - etc.
    :param output_directory: Output directory to store the results
    :return: list of (input_file, output_file) filename tuples.
    """

    subject_ids = os.listdir(input_directory)
    filenames = []
    for sub in subject_ids:
        if not os.path.isdir(os.path.join(input_directory, sub)):
            continue
        time_points = os.listdir(os.path.join(input_directory, sub))

        for t in time_points:
            print(sub, t)
            if not os.path.isdir(os.path.join(input_directory, sub, t)):
                continue
            scalars = os.listdir(os.path.join(input_directory, sub, t))
            for sc in scalars:
                if not os.path.isdir(os.path.join(input_directory, sub, t, sc)):
                    continue
                Path(os.path.join(output_directory, sub, t, sc)).mkdir(exist_ok=True, parents=True)
                for file in os.listdir(os.path.join(input_directory, sub, t, sc)):
                    if file.endswith("txt") and \
                            not os.path.isdir(
                                os.path.join(input_directory, sub, t, sc, file)
                            ):
                        input_file = os.path.join(input_directory, sub, t, sc, file)
                        output_file = os.path.join(
                            output_directory, sub, t, sc,
                            file.split(".")[0] + "_flat.jpeg")
                        filenames.append((input_file, output_file))
    return filenames


def run_single_subject(args):
    filenames = []
    sub = args["sub_dir"]
    if args["scalar"] is None:
        scalars = os.listdir(sub)
    else:
        scalars = [args["scalar"]]
    for sc in scalars:
        if not os.path.isdir(os.path.join(sub, sc)):
            continue
        for file in os.listdir(os.path.join(sub, sc)):
            if file.endswith("txt") and \
                    not os.path.isdir(
                        os.path.join(sub, sc, file)
                    ):
                input_file = os.path.join(sub, sc, file)
                output_file = os.path.join(sub, sc,
                    file.split(".")[0] + "_flat.jpeg")
                arguments = {
                    "feature_map": input_file,
                    "output": output_file,
                    "sphere_template": args["template"],
                    "scalar": sc,
                    "resolution": args["resolution"]
                }
                sgim_sampling_wrapper(arguments)
    return filenames


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--sub_dir", type=str, required=True,
                        help="Directory of the subjects")
    parser.add_argument("-s", "--template", type=str, required=True,
                        help="Sphere template")
    parser.add_argument("-r", "--resolution", type=str, default="512",
                        help="Resolution of output 2D image")
    parser.add_argument("--scalar", default=None, nargs="?",
                        help="Which scalar to process")
    args = vars(parser.parse_args())
    args["PYTHON_PATH"] = "/Users/mturja/PycharmProjects/geometry_image/geom_ccn_env/bin/python3"
    # run_single_subject(args)
    run_geom_image(args)
    print("======= End =======")


