import os.path

from flask import Flask, render_template, request, jsonify, make_response

from Up_to_Domain.interfaces import up_to_domain
from ascensionpoint_generate.interfaces import interface_visionpoint
from flask_cors import CORS

from config.params import *
from village_space_quantization.interface import save_raw_quantization_data
from config.params import run_mode

app = Flask(__name__)
# app = Flask(__name__, template_folder="static/templates", static_folder="static", static_url_path="") #暂未使用
app.config.from_object(__name__)
CORS(app, resources={r'/*': {'origins': '*'}}, supports_credentials=True)
app.jinja_env.variable_start_string = '{['
app.jinja_env.variable_end_string = ']}'

# app.config["SEND_FILE_MAX_AGE_DEFAULT"] = timedelta(seconds=1.0)

# 部署时注释掉
# @app.route("/", methods=['GET'])
# def mainPage():
#     return render_template('index.html')

#
# @app.route("/autoSortingPage", methods=['GET'])
# def autoSortingPage():
#     return render_template('autoSortingPage2.html')
#
#
# 调试时使用
# @app.route("/test", methods=['GET', 'POST'])
# def test():
#     return make_response(jsonify({
#         'result': {'msg': 'Success',
#                    'code': 0
#                    }
#     }))


@app.route("/api/uploadFiles", methods=['GET', 'POST'])
def uploadFiles():
    upload_files = request.files.getlist('files')
    global cur_file_name

    for upload_file in upload_files:
        if upload_file.filename.endswith('.tif'):
            dem_file = upload_file
        else:
            remote_file = upload_file
            cur_file_name = upload_file.filename.split('.')[0]
        upload_file.save('./static/datasets/' + upload_file.filename)

    dem_path = './static/datasets/' + dem_file.filename
    remotePath = './static/datasets/' + remote_file.filename

    path1, path2, centerCoord,location = up_to_domain.interface_drawRawData(dem_path, remotePath)
    print(centerCoord)
    result = make_response(jsonify({
        'result': {'dem_url': path1,
                   'remote_url': path2,
                   'centerCoord': centerCoord,
                   'location':location,
                   'code': 0},
    }))
    return result


@app.route("/api/distantCalc", methods=['GET', 'POST'])
def distantCalc():
    # global cur_file_name
    global timelimit
    global demAxis_use

    dis = request.json['time_limit']
    demAxis = request.json['demAxis']

    dis = int(dis)
    demAxis = float(demAxis)

    timelimit = dis
    demAxis_use = demAxis

    dem_path = './static/datasets/' + cur_file_name + '.tif'
    remotePath = './static/datasets/' + cur_file_name + '.png'

    centerpoint, accessdomain_path = up_to_domain.interface_uptodomain(dem_path, remotePath, dis, demAxis=30)

    # if isinstance(accesspoints,np.ndarray):
    #     accesspointsUse = accesspoints.tolist()
    # else:
    #     accesspointsUse = accesspoints

    # Only get the accessdomain points in format longtitude/latitude
    result = make_response(jsonify({
        'result': {'accessdomain_path': accessdomain_path,
                   'centerpoint': centerpoint,
                   'code': 0},
    }))
    return result


@app.route("/api/calAscensionPoint", methods=['GET', 'POST'])
def calAscensionPoint():
    demPath = './static/datasets/' + cur_file_name + '.tif'

    demAxis = demAxis_use

    ascensionPointNum = int(request.json['ascensionPointNum'])
    divideSpace = float(request.json['divideSpace'])

    path1, path2 = interface_visionpoint.interface_cal_ascensionpoint(demPath, ascensionPointNum, timelimit,
                                                                      divideSpace, demAxis)

    result = make_response(jsonify({
        'result': {'ascension_point_list': path1,
                   'ascensionpoint_viewshedpathlist': path2,
                   'code': 0},
    }))

    return result


@app.route("/api/calExposivePoint", methods=['GET', 'POST'])
def calExposivePoint():
    global cur_file_name
    global timelimit

    dem_path = './static/datasets/' + cur_file_name + '.tif'
    exposivePointNum = int(request.json['exposivePointNum'])

    exposivepoint_numlist_absolute, exposive_result_path = interface_visionpoint.interface_cal_exposivepoint(
        dem_path,
        timelimit,
        exposivePointNum)

    result = make_response(jsonify({
        'result': {'exposivepoint_numlist_absolute': exposivepoint_numlist_absolute,
                   'exposive_result_path': exposive_result_path,
                   'code': 0},
    }))

    return result


# wy special
@app.route("/wyFunction1", methods=["GET", "POST"])
def wyFunction1():
    return False


# Load all village locations once
@app.route("/api/getTxtCoordinate", methods=['GET', 'POST'])
def getTxtCoordinate():
    with open('static/datasets/village_metadata.txt') as file:
        content = file.readlines()

    CoordinateDict = {}

    for line in content:
        str_list = line.split()
        LngLat = (str_list[1], str_list[2])
        CoordinateDict[str_list[0]] = LngLat

    result = make_response(jsonify({
        'result': CoordinateDict,
        'code': 0
    }))
    return result


# Part of village space quantization
@app.route("/api/upLoadQuantizationFiles", methods=['GET', 'POST'])
def upLoadQuantizationFiles():
    upload_files = request.files.getlist('files')

    if not os.path.exists(VILLAGE_SPACE_QUANTIZATION):
        os.makedirs(VILLAGE_SPACE_QUANTIZATION)

    remote_path = ""
    mask_path = ""
    for upload_file in upload_files:
        if upload_file.filename.find("remote") != -1:
            remote_file = upload_file
            remote_path = os.path.join(VILLAGE_SPACE_QUANTIZATION, remote_file.filename)
        elif upload_file.filename.find("mask") != -1:
            mask_file = upload_file
            mask_path = os.path.join(VILLAGE_SPACE_QUANTIZATION, mask_file.filename)
        upload_file.save(VILLAGE_SPACE_QUANTIZATION + upload_file.filename)

    remote, mask = save_raw_quantization_data.save_raw_quantization(remote_path, mask_path)

    result = make_response(jsonify({
        'result': {'remote_url': remote,
                   'mask_url': mask,
                   'code': 0},
    }))
    return result


@app.route("/api/analysisQuantizationModel", methods=['GET', 'POST'])
def analysisQuantizationModel():
    from village_space_quantization.interface import interface_quantization
    mode = request.json["analysis_model"]
    mask_file = request.json["raw_mask"]

    mask_file_name = os.path.basename(mask_file).split('.')[0]

    if mode == "空间结构":
        result_path = interface_quantization.interface_analysis_space(mask_file_name)

        result = make_response(jsonify({
            'result': {'space_result': result_path,
                       'code': 0},
        }))
        return result
    elif mode == "边界形状":
        result_path = interface_quantization.interface_analysis_shape(mask_file_name)

        result = make_response(jsonify({
            'result': {'shape_result': result_path,
                       'code': 0},
        }))
        return result
    elif mode == "建筑秩序":

        result_path = interface_quantization.interface_analysis_order(mask_file_name)

        result = make_response(jsonify({
            'result': {'order_result': result_path,
                       'code': 0},
        }))

        return result

@app.route("/api/calRoadIntersection", methods=['GET', 'POST'])
def calRoadIntersection():
    from road_intersection.interfaces.interface_road_intersection import interface_get_road_intersection
    remote_path = './static/datasets/' + cur_file_name + '.png'

    _, road_intersection_path = interface_get_road_intersection(remote_path)

    result = make_response(jsonify({
        'result': {'roadIntersection_path': road_intersection_path,
                   'code': 0},
    }))

    return result

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # app.run(port=8000)
    # app.run(debug=True, port=8000, host='127.0.0.1')

    from commonutils.tools import Tools
    #
    Tools.clear_cache()
    Tools.clear_result(0)
