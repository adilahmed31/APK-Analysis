import yaml
datafile = "all_apk_data.yaml"

def permission_analysis(data):
    permission_stats = {
        "third_party": dict(),
        "top": dict(),
        "offstore": dict(),
        "dualuse": dict()
    }
    default_permission = ["android.hardware", "android.permission", "com.google", "com.android"]
    apk_types = {"top": 0, "offstore": 0, "dualuse": 0}
    apk_types_permissions = {"top": 0, "offstore": 0, "dualuse": 0}

    for key in data:
        # print(key)
        app = data[key]
        ipv = 0
        benign = 0
        dualuse = 0
        label = app["class"]
        apk = app["apk"]
        apk_type = ''
        for i in apk_types:
            if i in apk:
                apk_type = i
                # print(apk_type)
                apk_types[apk_type] += 1
        
        # print(label)
        if "permissions" in app:
            permissions = app["permissions"]

            for permission in permissions:
                third_party = True
                permission = permission.lower()

                if len(permission) == 0:
                    continue
                if permission[0] == ".":
                    third_party = False
                # print(apk_type)
                apk_types_permissions[apk_type] += 1
                # print(apk_types_permissions[apk_type])
                for i in default_permission:
                    if i in permission:
                        third_party = False
                
                if third_party:
                    if permission not in permission_stats["third_party"]:
                        permission_stats["third_party"][permission] = {
                            "total": 0,
                            "ipv": 0,
                            "benign": 0
                        }
                    permission_stats["third_party"][permission][label] += 1
                    permission_stats["third_party"][permission]["total"] += 1
                else:
                    if permission not in permission_stats[apk_type]:
                        permission_stats[apk_type][permission] = 1
                    else:
                        permission_stats[apk_type][permission] += 1
    # print(apk_types_permissions)
    for i in apk_types:
        apk_types_permissions[i] = float(apk_types_permissions[i] / apk_types[i])
    

    sorted_dict = dict()
    sorted_dict["Stats"] = apk_types_permissions
    sorted_dict["third_party"] = {k: v for k, v in sorted(permission_stats["third_party"].items(), key=lambda item: item[1]["ipv"], reverse=True)}
    for key in ["top", "offstore", "dualuse"]:
        sorted_dict[key] = {k: v for k, v in sorted(permission_stats[key].items(), key=lambda item: item[1], reverse=True)}
    return sorted_dict

def admin_analysis(data):  
    # print(data) 
    apk_types = ["top", "offstore", "dualuse"]
    keywords = ["admin", "accessibility"]
    admin_stats = dict()
    for app_type in apk_types:
        admin_stats[app_type] = dict()
        admin_stats[app_type]["Num"] = 1
        for keyword in keywords:
            admin_stats[app_type][keyword] = 0

    for key in data:
        # print(key)
        app = data[key]
        apk = app["apk"]
        label = app["class"]
        services = []
        receivers = []
        for i in apk_types:
            if i in apk:
                app_type = i
        admin_stats[app_type]["Num"] += 1
        if "services" in app:
            services = app["services"]
        if "receivers" in app:
            receivers = app["receivers"]
        
        keywords = {
            "admin": False,
            "accessibility": False
        }

        for item in receivers+services:
            # print(item)
            for keyword in keywords:
                if keyword in item:
                    keywords[keyword] = True
        
        for keyword in keywords:
            if keywords[keyword]:
                admin_stats[app_type][keyword] += 1
    return admin_stats

def get_package(data):
    data_list = data.split(".")
    if len(data_list) > 1:
        package = data_list[0] + "." + data_list[1]
        return package
    return data

def third_party_analysis(data):
    third_party = {}
    for key in data:
        app = data[key]
        package = get_package(key)
        services = []
        receivers = []
        label = app["class"]
        libs = []
        if "services" in app:
            services = app["services"]
        if "receivers" in app:
            receivers = app["receivers"]
        for receiver in receivers+services:
            
            if package not in receiver and receiver[0] != '.':
                lib_class = receiver.split(".")
                if len(lib_class) > 1:
                    lib_name = lib_class[0] + "." + lib_class[1]
                    
                    del lib_class[0]
                    del lib_class[0]
                    cl = ".".join(lib_class)
                    if lib_name not in libs:
                        libs.append(lib_name)
                    # print(lib_class)
                    # print(lib_name, cl)

                    if lib_name in third_party:
                        if cl in third_party[lib_name]:
                            third_party[lib_name][cl] += 1
                        else:
                            third_party[lib_name][cl] = 1
                    else:
                        third_party[lib_name] = dict()
                        third_party[lib_name]["ipv"] = 0
                        third_party[lib_name]["benign"] = 0
                        third_party[lib_name][cl] = 1
        for lib in libs:
            third_party[lib][label] += 1

    sorted_keys = sorted(third_party, key=lambda x: (third_party[x]['ipv']), reverse=True)

    sorted_dict = dict()
    sorted_dict["stats"] = {
        "ipv": {},
        "benign": {}
    }

    for key in sorted_keys:
        sorted_dict[key] = third_party[key]
        lib = third_party[key]
        if lib["ipv"] > 0: 
            sorted_dict["stats"]["ipv"][key] = lib["ipv"]

        if lib["benign"] > 0: 
            sorted_dict["stats"]["benign"][key] = lib["benign"]
    
    # sorted_dict = {k: third_party[k] for k in sorted_keys}

    return sorted_dict


def main():
    outfile_3rd_party_stats = "third_party_stats.yaml"
    Outfile_permission_stats = "permission_stats.yaml"
    outfile_admin_acessibility_stats = "admin_acessibility_stats.yaml"

    with open(datafile, 'r') as f: 
        data = yaml.load(f, Loader=yaml.FullLoader)
    third_party = third_party_analysis(data)
    with open(outfile_3rd_party_stats, 'w') as file:
        documents = yaml.dump(third_party, file, sort_keys=False)
    # print(data)
    admin_acessibility_stats = admin_analysis(data)
    with open(outfile_admin_acessibility_stats, 'w') as file:
        documents = yaml.dump(admin_acessibility_stats, file)

    permission_stats = permission_analysis(data)
    with open(Outfile_permission_stats, 'w') as file:
        documents = yaml.dump(permission_stats, file, sort_keys=False)

if __name__ == '__main__':
    main()
    