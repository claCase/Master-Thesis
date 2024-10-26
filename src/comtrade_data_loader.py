import urllib.error
import urllib.request
import urllib.parse
import pandas as pd
import numpy as np
import os
import json
import argparse as arg
import time
from bs4 import BeautifulSoup as bs
from itertools import product
import pickle as pkl
import eventlet
from eventlet.greenpool import GreenPool
import matplotlib.pyplot as plt
import tqdm

# import cpi


# cpi.update()
# os.chdir("../")
# print(os.getcwd())
COUNTRIES_CODES_PATH = os.path.join(
    os.getcwd(), "Comtrade", "Reference Table", "Comtrade Country Code and ISO list.xls"
)
CODES_CONVERSION_PATH = os.path.join(
    os.getcwd(),
    "Comtrade",
    "Reference Table",
    "CompleteCorrelationsOfHS-SITC-BEC_20170606.xls",
)
COMMODOTIES_FOLDER = os.path.join(
    os.getcwd(), "Comtrade", "Reference Table", "Commodity Classifications"
)
COMMODOTIES_REPORTING = [
    "H0",
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
    "HS",
    "S1",
    "S2",
    "S3",
    "S4",
    "ST",
]
COMMODOTIES_REPORTING4 = [
    "HS92",
    "HS96",
    "HS02",
    "HS07",
    "HS12",
    "HS17",
    "SITC1",
    "SITC2",
    "SITC3",
    "SITC4",
]
COMMODOTIES_REPORTING_COMBINED = dict()
for i, j in zip(COMMODOTIES_REPORTING, COMMODOTIES_REPORTING4):
    COMMODOTIES_REPORTING_COMBINED[i] = j

DATA_AVAILABILITY_PATH = os.path.join(os.getcwd(), "Data", "data_availability.pkl")
COMTRADE_URL = "https://comtrade.un.org/api/get?"
MONTHLY_DATA_PATH = os.path.join(os.getcwd(), "Data", "Monthly Data")
ANNUAL_DATA_PATH = os.path.join(os.getcwd(), "Data", "Annual Data")
COMTRADE_DATASET = os.path.join(os.getcwd(), "Data")
SEEN_PARAMS = os.path.join(os.getcwd(), "Data", "seen_params.pkl")
YEARS_AVAILABILITY = os.path.join(os.getcwd(), "Data", "years_availability.pkl")
PARAMS_LIST = os.path.join(os.getcwd(), "Data", "params_list.pkl")
COW_ISO3 = os.path.join(os.getcwd(), "Data", "cow2iso3.csv")
COW_CC = os.path.join(os.getcwd(), "Data", "cow2iso3.txt")


def load_countries_codes():
    cc_df = pd.read_excel(COUNTRIES_CODES_PATH)
    codes = cc_df["Country Code"]
    names = cc_df["Country Name, Abbreviation"]
    return codes, names


def get_countries_codes_from_web():
    URL = "https://www.statcan.gc.ca/eng/subjects/standard/sccai/2011/scountry-desc"
    path = os.path.join(os.getcwd(), "Data", "countries_codes.csv")
    if not os.path.exists(path):
        print(f"Loading countries codes from web {URL}")
        data = urllib.request.urlopen(URL).read()
        data = bs(data, "html.parser")

        trs = data.findAll("tr")
        rows = []
        for tr in trs:
            tds = tr.findAll("td")
            row = []
            for td in tds:
                text = td.getText()
                if text == "\xa0":
                    text = ""
                row.append(text)
            if len(row) == 7:
                rows.append(row)

        rows = rows[1:-1]
        rows = np.vstack(rows)
        df = pd.DataFrame(
            rows,
            columns=[
                "Country Name, Abbreviation",
                "num",
                "iso2",
                "iso3",
                "start",
                "end",
                "remarks",
            ],
        )
        # filter1 = df["Country Name, Abbreviation"].str.contains("Island")
        # df = df[~filter1]
        df.to_csv(path)
    else:
        print(f"Reading from saved file: {path}")
        df = pd.read_csv(path)
    return df


def filter_dataframe(dataframe):
    if "Country Name, Abbreviation" not in dataframe.columns:
        raise Exception(
            f"Column 'Country Name, Abbreviation' not in {dataframe.columns}"
        )
    exclude_countries = [
        "Africa CAMEU region, nes",
        "American Samoa",
        "Andorra",
        "Anguilla",
        "Antartica",
        "Antigua and Barbuda",
        "Areas, nes",
        "Aruba",
        "Bahamas",
        "Barbados",
        "Br. Antarctic Terr.",
        "Br. Indian Ocean Terr.",
        "Bunkers",
        "Cabo Verde",
        "CACM, nes",
        "Caribbean, nes",
        "Comoros",
        "Curaçao",
        "Djibouti",
        "Equatorial Guinea",
        "EU-28",
        "Europe EFTA, nes",
        "Europe EU, nes",
        "Fiji",
        "Fmr Panama, excl.Canal Zone",
        "Fmr Rhodesia Nyas",
        "Fmr Tanganyika",
        "Fmr Zanzibar and Pemba Isd",
        "Fr. South Antarctic Terr.",
        "Free Zones",
        "French Guiana",
        "French Polynesia",
        "FS Micronesia",
        "Guadeloupe",
        "Guam",
        "Haiti",
        "Holy See",
        "Honduras",
        "India, excl. Sikkim",
        "Kiribati",
        "LAIA, nes",
        "Lao People's Dem. Rep.",
        "Madagascar",
        "Maldives",
        "Malta",
        "Martinique",
        "Mauritius",
        "Mayotte",
        "Montserrat",
        "Nauru",
        "Neth. Antilles",
        "Neth. Antilles and Aruba",
        "Neutral Zone",
        "New Caledonia",
        "Niue",
        "North America and Central America, nes",
        "Northern Africa, nes",
        "Oceania, nes",
        "Other Africa, nes",
        "Other Asia, nes",
        "Other Europe, nes",
        "Palau",
        "Papua New Guinea",
        "Pitcairn",
        "Rest of America, nes",
        "Réunion",
        "Ryukyu Isd",
        "Sabah",
        "Saint Barthélemy",
        "Saint Helena",
        "Saint Kitts and Nevis",
        "Saint Kitts, Nevis and Anguilla",
        "Saint Lucia",
        "Saint Maarten",
        "Saint Pierre and Miquelon",
        "Saint Vincent and the Grenadines",
        "Samoa",
        "San Marino",
        "Sao Tome and Principe",
        "Seychelles",
        "Serbia and Montenegro",
        "So. African Customs Union",
        "Special Categories",
        "State of Palestine",
        "TFYR of Macedonia",
        "Timor-Leste",
        "Tokelau",
        "Tonga",
        "Trinidad and Tobago",
        "Tuvalu",
        "Vanuatu",
        "Western Asia, nes",
        "World",
        "Antarctica",
        "Macao SAR",
        "Eastern Europe",
        "nes Fmr Dem. Rep. of Vietnam",
        "Fmr Panama - Canal - Zone",
        "Bonaire",
        "Sikkim",
        "World",
    ]

    filter1 = dataframe["Country Name, Abbreviation"].str.contains("Island")
    filter2 = dataframe["Country Name, Abbreviation"].str.contains("Isds")
    filter3 = dataframe["Country Name, Abbreviation"].str.contains(
        "|".join(exclude_countries)
    )
    filtered = (~filter1) & (~filter2) & (~filter3)
    return filtered


def get_countries_codes_from_comtrade():
    df = pd.read_excel(COUNTRIES_CODES_PATH)
    filter = filter_dataframe(df)
    df = df[filter]
    return df


def codes_conversion_dict(code1: str, code2: str, precision=2, conversion_table=None):
    if conversion_table is None:
        print("Loading conversion table")
        conversion_table = pd.read_excel(
            CODES_CONVERSION_PATH,
            index_col=None,
            converters={
                0: str,
                1: str,
                2: str,
                3: str,
                4: str,
                5: str,
                6: str,
                7: str,
                8: str,
                9: str,
                10: str,
            },
        )
    conversion_table = conversion_table.dropna()
    codes = conversion_table.columns
    code1, code2 = [
        COMMODOTIES_REPORTING_COMBINED[code1],
        COMMODOTIES_REPORTING_COMBINED[code2],
    ]
    if code1 in codes and code2 in codes:
        codes_precision = [code + f"_p{str(precision)}" for code in [code1, code2]]
    else:
        raise Exception(f"Codes: {code1} {code2} not in {codes}")

    print("Making conversion dict")
    for code, code_2 in zip([code1, code2], codes_precision):
        column = conversion_table[code].astype(str).str[:precision].values
        conversion_table[code_2] = column

    # Get the most frequent conversion
    freq_conversion = conversion_table.groupby(
        [codes_precision[0], codes_precision[1]]
    ).size()
    freq_conversion = freq_conversion.reset_index()
    freq_conversion.columns = [*freq_conversion.columns[:-1], "Counts"]
    idx = (
        freq_conversion.groupby([codes_precision[0]])["Counts"].transform(
            lambda x: x.max()
        )
        == freq_conversion.Counts
    )
    freq_conversion = freq_conversion[idx]
    keys = freq_conversion[codes_precision[0]]
    values = freq_conversion[codes_precision[1]]
    return {k: v for k, v in zip(keys, values)}


def cow_iso3_to_country_codes(save=True):
    if not os.path.exists(COW_CC):
        mapping = {}
        with open(COW_ISO3, "r") as file:
            lines = file.splitlines()
            for line in lines:
                split = line.split(" ")
                idx = split[0]
                name = split[1]
                iso3 = split[2]
                iso2 = split[3]
                cwcode = split[4]

    else:
        with open(COW_CC, "rb") as file:
            mapping = pkl.load(file)
    return mapping


def get_data(
    country1: [str],
    country2: [str],
    commodity: [str],
    flow=(1,),
    frequency: str = "A",
    year: [str] = ("all",),
    month: [str] = ("",),
    reporting_code: str = "S1",
    proxy: {} = None,
    n_tries=0,
):
    commodity_ = "".join([str(c) + "," for c in commodity if c != ""])
    commodity_ = commodity_[:-1]
    flow_ = "".join([str(f) for f in flow])
    dates = []
    country1_ = "".join([str(country) + "," for country in country1 if country != ""])[
        :-1
    ]
    country2_ = "".join([str(country) + "," for country in country2 if country != ""])[
        :-1
    ]
    if month[0] != "" or year[0] != "all":
        for z, (i, j) in enumerate(product(year, month)):
            if z != len(year) * len(month):
                date = "".join([str(i), str(j), ","])[:-1]
            else:
                date = "".join([str(i), str(j)])
            dates.append(date)
        ps = ",".join(dates)
    else:
        ps = ",".join(year)

    values = {
        "max": 10000,
        "type": "C",
        "freq": frequency,
        "px": reporting_code,
        "ps": ps,
        "r": country1_,
        "p": country2_,
        "rg": flow_,
        "cc": commodity_,
    }
    data = "".join(
        [f"{key}={value}&" for key, value in zip(values.keys(), values.values())]
    )[:-1]
    # data = urllib.parse.urlencode(data)
    headers = {"User-Agent": "Chrome/35.0.1916.47"}
    req = urllib.request.Request(COMTRADE_URL + data, None, headers=headers)
    if proxy is not None:
        proxy_ = urllib.request.ProxyHandler(proxy)
        opener = urllib.request.build_opener(proxy_)
        try:
            print("Trying " + COMTRADE_URL + data + " + " + str(proxy))
            with opener.open(req, timeout=120) as response:
                data_response = response.read()
        except urllib.error.HTTPError as e:
            print(f"Download not successful because of {e} for proxy {proxy}")
            return [{"Error": 1}]

        except Exception as e:
            print(f"Download not successful because of {e} for proxy {proxy}")
            return [{"Error": 1}]
    else:
        print("Trying " + COMTRADE_URL + data)
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                data_response = response.read()
                eventlet.sleep(2)
        except urllib.error.HTTPError as e:
            if e.code == 409:
                print(f"HTTP error {e.code}")
                # return [{"Error":409}]
                raise urllib.error.HTTPError(
                    COMTRADE_URL + data, 409, "Conflict", headers, None
                )
        except Exception as e:
            print(f"Tried without proxy after and got error {e}")
            return [{"Error": 1}]

    data_parsed = json.loads(data_response)
    data_parsed = data_parsed["dataset"]
    # print(f"SUCCESS for {data} + {proxy}")
    return data_parsed


class AppURLopener(urllib.request.FancyURLopener):
    version = "Mozilla/5.0"


def scrape_proxy():
    URL = "https://www.sslproxies.org/"
    URL2 = "https://proxylist.geonode.com/api/proxy-list?limit=200&page=1&sort_by=lastChecked&sort_type=desc"

    list1 = []
    fancy = AppURLopener()
    with fancy.open(URL2) as resp:
        resp = resp.read()
    resp = json.loads(resp)["data"]
    for r in resp:
        if r["ip"]:
            list1.append({f"{r['protocols'][0]}": f"{r['ip']}:{r['port']}"})

    list2 = []
    req = urllib.request.Request(
        URL, None, headers={"User-Agent": "Chrome/35.0.1916.47 "}
    )
    resp = urllib.request.urlopen(req).read()
    doc = bs(resp, "html.parser")
    trs = doc.findAll("tr")
    for tr in trs:
        tds = tr.findAll("td")
        prox = {}
        prox_url = ""
        prox_port = ""
        sec = ""
        for i, td in enumerate(tds):
            if i == 0:
                prox_url = td.getText()
            if i == 1:
                prox_port = td.getText()
            if i == 6:
                prox_sec = td.getText()
                if prox_sec == "yes":
                    sec = "https"
                else:
                    sec = "http"
        prox[sec] = prox_url + "".join([":", str(prox_port)])
        list2.append(prox)
    list2 = list2[1:-8]
    with open("Data/proxies.txt", "r") as file:
        list3 = file.readlines()
    list3_dict = []
    for line in list3:
        l = line.split(" ")
        ip_port = l[0]
        if len(set(l[1].split("-"))) == 3:
            list3_dict.append({"https": ip_port})
        else:
            list3_dict.append({"http": ip_port})
    list1.extend(list3_dict)

    list1.extend(list2)
    return list1


def check_proxy_alive(proxy):
    proxy_ = urllib.request.ProxyHandler(proxy)
    opener = urllib.request.build_opener(proxy_)
    try:
        opener.open("https://comtrade.un.org/Data/", timeout=30).close()
    except Exception as e:
        print(e)
        return None
    else:
        print(proxy)
        return proxy


def parallelize_check_proxy(proxies: [dict]):
    avail_proxies = []
    eventlet.monkey_patch()
    pool = GreenPool(250)

    for proxy in pool.imap(check_proxy_alive, proxies):
        if proxy is not None:
            avail_proxies.append(proxy)
    return avail_proxies


def parallelize_get_data_requests(params):
    data_list = []
    eventlet.monkey_patch(thread=True)
    pool = eventlet.GreenPool(len(params))
    for data in pool.starmap(get_data, params):
        data_list.append(data)
    return data_list


def check_years_data_availability(codes=None, save=True):
    if codes is None:
        df = get_countries_codes_from_comtrade()
        codes = df["Country Code"].astype(str).to_numpy().flatten()
    proxies = scrape_proxy()
    proxies = parallelize_check_proxy(proxies)
    random_idx = np.random.choice(
        np.arange(0, len(proxies)), size=len(proxies), replace=False
    )
    proxies = [proxies[i] for i in random_idx]
    # Get already checked countries keys and initialize values as empty list
    if os.path.exists(DATA_AVAILABILITY_PATH):
        with open(DATA_AVAILABILITY_PATH, "rb") as file:
            print(f"Reading from {DATA_AVAILABILITY_PATH}")
            country_avail_dict = pkl.load(file)
            keys = country_avail_dict.keys()
            already_provided = set(keys)
            codes_final = []
            for i, c in enumerate(codes):
                if c not in already_provided:
                    country_avail_dict[str(c)] = []
                    codes_final.append(str(c))
            codes = np.asarray(codes_final)
    else:
        country_avail_dict = {}
        already_provided = set([])
        for code in codes:
            country_avail_dict[str(code)] = []

    # Make requests for 5 countries at time
    div_c, mod_c = divmod(len(codes), 5)
    if mod_c != 0:
        codes_pad = [""] * (5 - mod_c)
        codes = np.concatenate([codes, codes_pad])
        codes = codes.reshape(-1, 5)
    else:
        codes = codes.reshape(-1, 5)
    if len(proxies) < len(codes):
        div, mod = divmod(len(codes), len(proxies))
        tiles = div + 1 if mod != 0 else div
        assignments = np.tile(np.arange(0, len(proxies)), tiles)
        assignments = assignments[: len(codes)]
        assigned_proxies = [proxies[i] for i in assignments]
    else:
        assigned_proxies = proxies[: len(codes)]

    p = ["0"] * len(codes)
    cc = ["TOTAL"] * len(codes)
    flow = [1] * len(codes)
    freq = ["A"] * len(codes)
    yrs = ["all"] * len(codes)
    months = [""] * len(codes)
    rc = ["S1"] * len(codes)
    params = [
        (
            codes[i],
            [p[i]],
            [cc[i]],
            [flow[i]],
            freq[i],
            [yrs[i]],
            [months[i]],
            rc[i],
            assigned_proxies[i],
        )
        for i in range(0, len(codes))
    ]

    batches = len(codes) // len(proxies) + 1
    batch_size = len(proxies)
    for t in range(0, batches):
        print(f"Batch {t}")
        if t < batches:
            batch_params = params[t * batch_size : (t + 1) * batch_size]
        else:
            batch_params = params[t * batch_size :]
        try:
            # time.sleep(30)
            data_list = parallelize_get_data_requests(batch_params)
            # print(data_list)
            for data in data_list:
                for row in data:
                    row = dict(row)
                    if bool(row):
                        c = str(row["rtCode"])
                        yr = row["yr"]
                        country_avail_dict[c].extend([yr])
        except Exception as e:
            print(f"Comtrade Exception {e}")
        else:
            if save:
                # Removing keys with empty value
                remove = []
                for k, v in zip(country_avail_dict.keys(), country_avail_dict.values()):
                    if not v:
                        remove.append(k)
                for k in remove:
                    country_avail_dict.pop(k)
                with open(DATA_AVAILABILITY_PATH, "wb") as file:
                    pkl.dump(country_avail_dict, file)
        time.sleep(2)

    return country_avail_dict


def select_most_avail_countries(quantile=0.5):
    if os.path.exists(DATA_AVAILABILITY_PATH):
        with open(DATA_AVAILABILITY_PATH, "rb") as file:
            print(f"Reading from {DATA_AVAILABILITY_PATH}")
            country_avail_dict = pkl.load(file)
    else:
        raise Exception(f"Data Availability file not found in {DATA_AVAILABILITY_PATH}")

    yrs = []
    for k, v in zip(country_avail_dict.keys(), country_avail_dict.values()):
        t = len(v)
        yrs.append(t)
    yrs = np.asarray(yrs)
    q = np.quantile(yrs, quantile)
    print(q)
    plt.hist(yrs, density=False)
    plt.plot((q, q), (0, quantile * len(country_avail_dict)), color="green")
    plt.show()
    df = get_countries_codes_from_comtrade()
    for c, y in zip(country_avail_dict.keys(), country_avail_dict.values()):
        if len(y) > 44:
            print(
                c,
                len(y),
                df[df["Country Code"] == int(c)]["Country Name, Full "].values[0],
            )


def link_func(
    gt,
    edge_list,
    param_idx,
    q,
    seen: set,
    data=None,
    q_proxy: eventlet.Queue = None,
    proxy=None,
    save_path=os.path.join(COMTRADE_DATASET, "complete_data.pkl"),
):
    error = False
    if gt is not None:
        data = gt.wait()

    elif data is None:
        raise Exception("Data cannot be None if no gt is available")
    row = {}
    for row in data:
        row = dict(row)
        if "Error" in row.keys():
            q.put(param_idx)
            error = True
            break
        elif bool(row):
            y = row["period"]
            c1 = row["rtCode"]
            c2 = row["ptCode"]
            p = row["cmdCode"]
            tv = row["TradeValue"]
            e = [y, c1, c2, str(p), tv]
            edge_list.append(e)
            seen.add(param_idx)

    if q_proxy is not None and not error:
        q_proxy.put(proxy)
    if q_proxy is not None and error:
        print(f"Removing from Queue Proxy {proxy}")
    if bool(row) and not error:
        with open(save_path, "wb") as file:
            pkl.dump(edge_list, file)
        with open(SEEN_PARAMS, "wb") as file:
            pkl.dump(seen, file)
            print(f"Q-size {q.qsize()}")
    elif not bool(row) and not error:
        print("Data empty")
        pass


def build_dataset(
    flow=("1",),
    frequency="A",
    reporting_code="SITC1",
    use_proxy=True,
    file_name="complete_data_new",
):
    if not os.path.exists(PARAMS_LIST):
        if os.path.exists(DATA_AVAILABILITY_PATH):
            with open(DATA_AVAILABILITY_PATH, "rb") as file:
                availability = pkl.load(file)
        else:
            availability = check_years_data_availability()
            raise Exception(f"No country data availability in {DATA_AVAILABILITY_PATH}")

        if os.path.exists(YEARS_AVAILABILITY):
            with open(YEARS_AVAILABILITY, "rb") as file:
                df_avail = pkl.load(file)
        else:
            df_avail = {}
            for i in range(1962, 2021):
                df_avail[str(i)] = []

            for country in availability.keys():
                yrs = availability[country]
                for y in yrs:
                    df_avail[str(y)].append(country)

            with open(YEARS_AVAILABILITY, "wb") as file:
                pkl.dump(df_avail, file)

        # s1_h1 = codes_conversion_dict("S1", "H1")
        conversion_table = pd.read_excel(
            CODES_CONVERSION_PATH,
            index_col=None,
            converters={
                0: str,
                1: str,
                2: str,
                3: str,
                4: str,
                5: str,
                6: str,
                7: str,
                8: str,
                9: str,
                10: str,
            },
        )
        s1 = list(
            set(conversion_table[reporting_code].dropna().astype(str).str[:2].to_list())
        )
        # s1 = [s for s in s1_h1.keys()]
        div_s, mod_s = divmod(len(s1), 19)
        s1_pad = [""] * (19 - mod_s)
        s1.extend(s1_pad)
        s1 = np.asarray(s1).reshape(-1, 19).tolist()

        for y, c in zip(df_avail.keys(), df_avail.values()):
            div_l, mod_l = divmod(len(c), 5)
            if mod_l:
                df_avail[y].extend([""] * (5 - mod_l))
            df_avail[y] = np.asarray(df_avail[y]).reshape(-1, 5).tolist()

        # Making list of param requests
        print("Making Params List")
        params = []
        for y in df_avail.keys():  # for each year
            len_c5 = len(df_avail[y])  # select available countries for year y
            for c5 in range(0, len_c5):
                countries = tuple(df_avail[y][c5])
                len_p = len(s1)
                for p in range(0, len_p):
                    products = tuple(s1[p])
                    params.append(
                        [
                            countries,
                            ("all",),
                            products,
                            tuple(flow),
                            frequency,
                            (y,),
                            ("",),
                            "S1",
                        ]
                    )
        with open(PARAMS_LIST, "wb") as file:
            print("Saving Params List")
            pkl.dump(params, file)
    else:
        with open(PARAMS_LIST, "rb") as file:
            print("Loading Params List")
            params = pkl.load(file)
    if not os.path.exists(SEEN_PARAMS):
        open(SEEN_PARAMS, "a").close()
    if os.path.getsize(SEEN_PARAMS) > 0:
        with open(SEEN_PARAMS, "rb") as seen_params:
            seen = pkl.load(seen_params)
            params_idx = np.arange(0, len(params))
            params_idx = np.delete(params_idx, list(seen))
            print(f"Total params to spawn: {len(params_idx)}")
    else:
        seen = set()
        params_idx = np.arange(0, len(params))

    if not os.path.exists(os.path.join(COMTRADE_DATASET, file_name + f"_{flow}.pkl")):
        open(os.path.join(COMTRADE_DATASET, file_name + f"_{flow}.pkl"), "a").close()
    if os.path.getsize(os.path.join(COMTRADE_DATASET, file_name + f"_{flow}.pkl")) > 0:
        with open(
            os.path.join(COMTRADE_DATASET, file_name + f"_{flow}.pkl"), "rb"
        ) as file:
            edge_list = pkl.load(file)
    else:
        edge_list = []

    # ___________ Initialize Batches ___________
    threads = 200
    pool = eventlet.GreenPool(threads)
    q = eventlet.Queue()
    q_proxy = eventlet.Queue()
    for idx in params_idx:
        q.put(idx)

    time_to_check = time.time()
    time_to_check_proxy = time.time()

    save_path = os.path.join(COMTRADE_DATASET, file_name + f"_{flow}.pkl")

    while not q.empty():
        if os.path.getsize(save_path) > 60e6:
            i = 0
            new_file = save_path[:-4] + f"_{i}.pkl"
            while os.path.exists(new_file):
                i += 1
                new_file = save_path[:-4] + f"_{i}.pkl"
            print(f"creating new file {new_file}")
            open(new_file, "a").close()
            save_path = new_file
            edge_list = []

        if use_proxy:
            t = time.time()
            if t > time_to_check_proxy or q_proxy.empty():
                proxies = scrape_proxy()
                proxies = parallelize_check_proxy(proxies)
                for proxy in proxies:
                    q_proxy.put(proxy)
                time_to_check = 60 * 60 + time.time()

            batch_counter = 0

            proxy = q_proxy.get()
            idx = q.get()
            print("Spawning proxy")
            thread_param = (*params[idx], proxy)
            thread = pool.spawn(get_data, *thread_param)
            thread.link(
                link_func,
                edge_list,
                idx,
                q,
                seen,
                q_proxy=q_proxy,
                proxy=proxy,
                save_path=save_path,
            )
            batch_counter += 1

            if not batch_counter % threads:
                pool.waitall()

        t = time.time()
        if t > time_to_check:
            try:
                print("Spawning without proxy")
                idx = q.get()
                thread_param = (*params[idx], None)
                thread = pool.spawn(get_data, *thread_param)
                result = thread.wait()
            except urllib.error.HTTPError as e:
                if e.code == 409:
                    print(f"Cannot Use My Machine because of Exception {e}")
                    time_to_check = 60 * 60 + time.time()
                    q.put(idx)
            except Exception as e:
                print(f"Trying with my machine Exception {e}")
                q.put(idx)
            else:
                link_func(
                    None,
                    edge_list=edge_list,
                    param_idx=idx,
                    q=q,
                    seen=seen,
                    data=result,
                    save_path=save_path,
                )
    print("Queue empty")


def join_pickles(file_name_match, folder):
    import re

    data = []
    for dirs, folders, files in os.walk(os.path.join(folder)):
        for file in files:
            if re.match(file_name_match, file):
                print(f"filename: {file}")
                with open(os.path.join(folder, file), "rb") as f:
                    d = pkl.load(f)
                    data.extend(d)
    with open(
        os.path.join(folder, file_name_match + "_joined_complete.pkl"), "wb"
    ) as f:
        pkl.dump(data, f)


def products_to_idx(reporting_code="SITC1"):
    if not os.path.exists(f"./Data/{reporting_code}_to_idx.pkl"):
        conversion_table = pd.read_excel(
            CODES_CONVERSION_PATH,
            index_col=None,
            converters={
                0: str,
                1: str,
                2: str,
                3: str,
                4: str,
                5: str,
                6: str,
                7: str,
                8: str,
                9: str,
                10: str,
            },
        )
        s1 = list(
            set(conversion_table[reporting_code].dropna().astype(str).str[:2].to_list())
        )
        idx = np.arange(len(s1))
        mapping = {}
        for i, j in zip(s1, idx):
            mapping[i] = j

        with open(f"./Data/{reporting_code}_to_idx.pkl", "wb") as file:
            pkl.dump(mapping, file)
    else:
        with open(f"./Data/{reporting_code}_to_idx.pkl", "rb") as file:
            mapping = pkl.load(file)
    return mapping


def countries_to_idx():
    if not os.path.exists(os.path.join(COMTRADE_DATASET, "countries_to_idx.pkl")):
        countries = get_countries_codes_from_comtrade()
        codes = countries["Country Code"].astype(int).to_numpy().flatten()
        idx = np.arange(len(codes))
        conversion = {}
        for i, j in zip(codes, idx):
            conversion[i] = j
        with open(os.path.join(COMTRADE_DATASET, "countries_to_idx.pkl"), "wb") as file:
            pkl.dump(conversion, file)
    else:
        with open(os.path.join(COMTRADE_DATASET, "countries_to_idx.pkl"), "rb") as file:
            conversion = pkl.load(file)
    return conversion


def product_to_idx(data, dict, data_path=None, conversion_dict_path=None):
    if data_path is not None:
        with open(data_path, "rb") as file:
            data = pkl.load(file)
    else:
        assert data is not None
        data = data
    if conversion_dict_path is not None:
        with open(conversion_dict_path, "rb") as file:
            idx_conversion = pkl.load(file)
    else:
        idx_conversion = dict
    no_edge = []
    for i, edge in enumerate(data):
        try:
            data[i] = (*edge[:-2], idx_conversion[edge[-2]], edge[-1])
        except:
            print(f"No key for {edge[-2]}")
            no_edge.append((i, edge[-2]))
    for i, e in enumerate(no_edge):
        data.pop(i)
    return data


def idx_to_product(data=None, reporting_code="SITC1"):
    if os.path.exists(
        os.path.join(COMTRADE_DATASET, f"idx_to_prod_{reporting_code}.pkl")
    ):
        with open(
            os.path.join(COMTRADE_DATASET, f"idx_to_prod_{reporting_code}.pkl"), "rb"
        ) as file:
            idx_to_prod = pkl.load(file)
    else:
        prod_to_idx = products_to_idx(reporting_code)
        idx_to_prod = {}
        for k in prod_to_idx.keys():
            idx_to_prod[prod_to_idx[k]] = k
        with open(
            os.path.join(COMTRADE_DATASET, f"idx_to_prod_{reporting_code}.pkl"), "wb"
        ) as file:
            pkl.dump(idx_to_prod, file)
    idxs = []
    if data is not None:
        for i in data:
            idxs.append(idx_to_prod[i])
        return idxs
    else:
        return idx_to_prod


def data_countries_to_idx(data, dict, data_path=None, conversion_dict_path=None):
    if data_path is not None:
        with open(data_path, "rb") as file:
            data = pkl.load(file)
    else:
        data = data

    if conversion_dict_path is not None:
        with open(conversion_dict_path, "rb") as file:
            idx_conversion = pkl.load(file)
    else:
        idx_conversion = dict

    no_key = []
    counter = tqdm.tqdm(total=len(data), desc="Converting countries codes")
    for i, e in enumerate(data):
        try:
            c1 = idx_conversion[e[1]]
            c2 = idx_conversion[e[2]]
            data[i] = (e[0], c1, c2, *e[3:])
        except:
            print(f"no key for {e}")
            no_key.append(i)
        counter.update(1)

    for i in no_key:
        data.pop(i)
    return data


def idx_to_countries(data=None):
    if os.path.exists(os.path.join(COMTRADE_DATASET, "idx_to_countries.pkl")):
        with open(os.path.join(COMTRADE_DATASET, "idx_to_countries.pkl"), "rb") as file:
            idx_to_country = pkl.load(file)
    else:
        country_to_idx = countries_to_idx()
        idx_to_country = {}
        for k in country_to_idx.keys():
            idx_to_country[country_to_idx[k]] = k
            with open(
                os.path.join(COMTRADE_DATASET, "idx_to_countries.pkl"), "wb"
            ) as file:
                pkl.dump(idx_to_country, file)
    converted = []
    if data is not None:
        for i in data:
            converted.append(idx_to_country[i])
        return converted
    else:
        return idx_to_country


def filter_not_in_countries_codes(data, file_name=None):
    print("Filtering world partners codes")
    not_in_cc = []
    countries_codes = countries_to_idx().keys()
    counter = tqdm.tqdm(total=len(data), desc="Getting world edges")
    for i, e in enumerate(data):
        if str(e[2]) not in countries_codes:
            not_in_cc.append(i)
        counter.update(1)

    counter = tqdm.tqdm(total=len(not_in_cc), desc="Popping world edges")
    for i in sorted(not_in_cc, reverse=True):
        del data[i]
        counter.update(1)

    if file_name:
        with open(os.path.join(COMTRADE_DATASET, file_name), "wb") as file:
            pkl.dump(data, file)
    return data, not_in_cc


def data_to_idx(
    data, countries_conversion_dict=None, product_conversion_dict=None, filename=None
):
    if countries_conversion_dict is None:
        country_dict = countries_to_idx()
    else:
        country_dict = countries_conversion_dict
    if product_conversion_dict is None:
        product_dict = products_to_idx()
    else:
        product_dict = product_conversion_dict

    counter = tqdm.tqdm(total=len(data), desc="Codes to idx conversion")
    no_key = []
    for i, e in enumerate(data):
        if e[2] != 0:
            try:
                c1 = country_dict[str(e[1])]
                c2 = country_dict[str(e[2])]
                p = product_dict[str(e[3])]
                data[i] = (e[0], p, c1, c2, e[4])
            except:
                no_key.append(i)
        else:
            no_key.append(i)
        counter.update(1)

    for i in sorted(no_key, reverse=True):
        data.pop(i)

    if filename is not None:
        with open(os.path.join(COMTRADE_DATASET, filename), "wb") as file:
            pkl.dump(data, file)
    return data, no_key


def check_and_eliminate_duplicate(data):
    duplicate = {}
    t = tqdm.tqdm(total=len(data))
    for i, e in enumerate(data):
        try:
            duplicate[str(e[:4])].append(i)
        except:
            duplicate[str(e[:4])] = [i]
        t.update(1)

    duplicate_idx = []
    for k in duplicate.keys():
        if len(duplicate[k]) > 1:
            duplicate_idx.append(duplicate[k][0])
    del duplicate
    data_np = np.asarray(data, dtype=np.float32)
    data_np = np.delete(data_np, duplicate_idx, 0)
    return data_np


def iso3_to_country_name(iso3_list=None):
    save_dir = os.path.join(COMTRADE_DATASET, "iso3_to_name.pkl")
    if os.path.exists(save_dir):
        with open(save_dir, "rb") as file:
            name_dict = pkl.load(file)
    else:
        name_dict = {}
        codes, names = load_countries_codes()
        for c, name in zip(codes, names):
            name_dict[c] = name
        with open(save_dir, "wb") as file:
            pkl.dump(name_dict, file)
    if iso3_list is None:
        return name_dict
    else:
        converted_names = []
        for i in iso3_list:
            converted_names.append(name_dict[i])
        return converted_names


if __name__ == "__main__":
    parser = arg.ArgumentParser()
    parser.add_argument(
        "-month", action="store_true", help="Monthly or Annual, default = Annual"
    )
    parser.add_argument(
        "--flow", type=int, default=1, help="1 for import, 2 for export"
    )
    parser.add_argument("--proxy", action="store_true", help="Use proxy")

    args = parser.parse_args()
    monthly = args.month
    proxy = args.proxy
    flow = args.flow
    # COMTRADE_DATASET = os.path.join(os.getcwd(), "Data")
    SEEN_PARAMS = os.path.join(os.getcwd(), "Data", f"seen_params_{flow}.pkl")

    if monthly:
        save_path = MONTHLY_DATA_PATH
        frequency = "M"
    else:
        save_path = ANNUAL_DATA_PATH
        frequency = "A"
    # load_countries_codes()
    build_dataset(flow=[str(flow)], use_proxy=proxy)
