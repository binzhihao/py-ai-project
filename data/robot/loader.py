import os
import pandas as pd

module_path = os.path.dirname(__file__)


def load_robot_execution(path=module_path, multiclass=False):
    with open(os.path.join(path, 'lp1.data')) as f:
        cur_id = 0
        time = 0

        id_to_target = {}
        data_rows = []

        for line in f.readlines():
            if line[0] not in ['\t', '\n']:
                cur_id += 1
                time = 0
                if multiclass:
                    id_to_target[cur_id] = line.strip()
                else:
                    id_to_target[cur_id] = 1 if (line.strip() == 'normal') else 0
            elif line[0] == '\t':
                values = list(map(int, line.split('\t')[1:]))
                data_rows.append([cur_id, time] + values)
                time += 1

        df = pd.DataFrame(data_rows, columns=['id', 'time', 'F_x', 'F_y', 'F_z', 'T_x', 'T_y', 'T_z'])
        target = pd.Series(id_to_target)

    return df, target
