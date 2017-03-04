import json
from phys_tools import models

MODELS = {
    "<class 'phys_tools.models.odor.OdorSession'>": models.OdorSession,
    "<class 'phys_tools.models.odor.PatternSession'>": models.PatternSession
}


def save_units_json(json_path: str, units: list):
    """
    Saves JSON format

    :param json_path: path to make file
    :param units: list of unit models to save
    :return:
    """
    units.sort()  # this will group the units in order by session.
    json_dict = {}
    last_s = None
    for unit in units:  # type: models.Unit
        session = unit.session
        if session != last_s:
            sess_id = str(session)
            s_type = str(type(session))
            assert s_type in MODELS.keys(), "Sorry this model type is not specified in the MODELS dict."
            json_dict[sess_id] = {
                'filenames': session.paths,
                'sessionModelType': s_type,
                'unitIDs': []
            }
            last_s = session
        _, unum = str(unit).split('u')  # only save the unit number.
        json_dict[sess_id]['unitIDs'].append(int(unum))
    with open(json_path, 'w') as f:
        json.dump(json_dict, f, indent="\t", )


def load_units_json(json_path: str):
    """
    Loads a JSON with subset of units.

    :param json_path: path
    :return: tuple(units, sessions)
    """
    with open(json_path, 'r') as f:
        json_dict = json.load(f)  # parse int doesn't work when ints are embedded
    all_units = []
    all_sessions = []
    for s_name, s_val in json_dict.items():
        fns = s_val['filenames']
        datfn = fns['dat']
        s_type_name = s_val['sessionModelType']
        SType = MODELS[s_type_name]
        u_ids = s_val['unitIDs']  # list of ints

        s = SType(datfn)
        s.set_unit_subset(u_ids)  # type: list
        all_units.extend(s.units(u_ids))
        all_sessions.append(s)
    return all_sessions


