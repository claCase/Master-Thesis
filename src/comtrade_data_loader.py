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
os.chdir("../")

COUNTRIES_CODES_PATH = os.path.join(os.getcwd(), "Comtrade", "Reference Table",
                                    "Comtrade Country Code and ISO list.xls")
CODES_CONVERSION_PATH = os.path.join(os.getcwd(), "Comtrade", "Reference Table",
                                     "CompleteCorrelationsOfHS-SITC-BEC_20170606.xls")
COMMODOTIES_FOLDER = os.path.join(os.getcwd(), "Comtrade", "Reference Table", "Commodity Classifications")
COMMODOTIES_REPORTING = ["H0", "H1", "H2", "H3", "H4", "H5", "HS", "S1", "S2", "S3", "S4", "ST"]
COMMODOTIES_REPORTING4 = ["HS92", "HS96", "HS02", "HS07", "HS12", "HS17", "SITC1", "SITC2", "SITC3", "SITC4"]
COMMODOTIES_REPORTING_COMBINED = dict()
for i, j in zip(COMMODOTIES_REPORTING, COMMODOTIES_REPORTING4):
    COMMODOTIES_REPORTING_COMBINED[i] = j

DATA_AVAILABILITY_PATH = os.path.join(os.getcwd(), "Data", "data_availability.pkl")
COMTRADE_URL = 'https://comtrade.un.org/api/get?'
MONTHLY_DATA_PATH = os.path.join(os.getcwd(), "Data", "Monthly Data")
ANNUAL_DATA_PATH = os.path.join(os.getcwd(), "Data", "Annual Data")
COMTRADE_DATASET = os.path.join(os.getcwd(), "Data", "complete_data.pkl")
SEEN_PARAMS = os.path.join(os.getcwd(), "Data", "seen_params.pkl")
YEARS_AVAILABILITY = os.path.join(os.getcwd(), "Data", "years_availability.pkl")
PARAMS_LIST = os.path.join(os.getcwd(), "Data", "params_list.pkl")


def load_countries_codes():
    cc_df = pd.read_excel(COUNTRIES_CODES_PATH)
    codes = cc_df["ctyCode"]
    names = cc_df["cty Name English"]
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
        df = pd.DataFrame(rows,
                          columns=["Country Name, Abbreviation", "num", "iso2", "iso3", "start", "end", "remarks"])
        # filter1 = df["Country Name, Abbreviation"].str.contains("Island")
        # df = df[~filter1]
        df.to_csv(path)
    else:
        print(f"Reading from saved file: {path}")
        df = pd.read_csv(path)
    return df


def filter_dataframe(dataframe):
    if "Country Name, Abbreviation" not in dataframe.columns:
        raise Exception(f"Column 'Country Name, Abbreviation' not in {dataframe.columns}")
    exclude_countries = ["Africa CAMEU region, nes", "American Samoa", "Andorra", "Anguilla", "Antartica",
                         "Antigua and Barbuda", "Areas, nes", "Aruba", "Bahamas", "Barbados", "Br. Antarctic Terr.",
                         "Br. Indian Ocean Terr.", "Bunkers", "Cabo Verde", "CACM, nes", "Caribbean, nes", "Comoros",
                         "Curaçao", "Djibouti", "Equatorial Guinea", "EU-28", "Europe EFTA, nes", "Europe EU, nes",
                         "Fiji", "Fmr Panama, excl.Canal Zone", "Fmr Rhodesia Nyas", "Fmr Tanganyika",
                         "Fmr Zanzibar and Pemba Isd", "Fr. South Antarctic Terr.", "Free Zones", "French Guiana",
                         "French Polynesia", "FS Micronesia", "Guadeloupe", "Guam", "Haiti",
                         "Holy See", "Honduras", "India, excl. Sikkim", "Kiribati", "LAIA, nes",
                         "Lao People's Dem. Rep.", "Madagascar", "Maldives", "Malta", "Martinique", "Mauritius",
                         "Mayotte", "Montserrat", "Nauru", "Neth. Antilles", "Neth. Antilles and Aruba", "Neutral Zone",
                         "New Caledonia", "Niue", "North America and Central America, nes", "Northern Africa, nes",
                         "Oceania, nes", "Other Africa, nes", "Other Asia, nes", "Other Europe, nes", "Palau",
                         "Papua New Guinea", "Pitcairn", "Rest of America, nes", "Réunion", "Ryukyu Isd", "Sabah",
                         "Saint Barthélemy", "Saint Helena", "Saint Kitts and Nevis", "Saint Kitts, Nevis and Anguilla",
                         'Saint Lucia', 'Saint Maarten', 'Saint Pierre and Miquelon',
                         'Saint Vincent and the Grenadines',
                         'Samoa', 'San Marino', 'Sao Tome and Principe', "Seychelles", "Serbia and Montenegro",
                         "So. African Customs Union", "Special Categories", "State of Palestine", "TFYR of Macedonia",
                         'Timor-Leste', "Tokelau", "Tonga", "Trinidad and Tobago", "Tuvalu", "Vanuatu",
                         "Western Asia, nes", "World", "Antarctica", "China", "Macao SAR", "Eastern Europe",
                         "nes Fmr Dem. Rep. of Vietnam", "Fmr Panama - Canal - Zone", "Bonaire", "Sikkim", "World"]

    filter1 = dataframe["Country Name, Abbreviation"].str.contains("Island")
    filter2 = dataframe["Country Name, Abbreviation"].str.contains("Isds")
    filter3 = dataframe["Country Name, Abbreviation"].str.contains("|".join(exclude_countries))
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
        conversion_table = pd.read_excel(CODES_CONVERSION_PATH, index_col=None,
                                         converters={0: str, 1: str, 2: str, 3: str, 4: str, 5: str, 6: str, 7: str,
                                                     8: str, 9: str, 10: str})
    conversion_table = conversion_table.dropna()
    codes = conversion_table.columns
    code1, code2 = [COMMODOTIES_REPORTING_COMBINED[code1], COMMODOTIES_REPORTING_COMBINED[code2]]
    if code1 in codes and code2 in codes:
        codes_precision = [code + f"_p{str(precision)}" for code in [code1, code2]]
    else:
        raise Exception(f"Codes: {code1} {code2} not in {codes}")

    print("Making conversion dict")
    for code, code_2 in zip([code1, code2], codes_precision):
        column = conversion_table[code].astype(str).str[:precision].values
        conversion_table[code_2] = column

    # Get the most frequent conversion
    freq_conversion = conversion_table.groupby([codes_precision[0], codes_precision[1]]).size()
    freq_conversion = freq_conversion.reset_index()
    freq_conversion.columns = [*freq_conversion.columns[:-1], "Counts"]
    idx = freq_conversion.groupby([codes_precision[0]])["Counts"].transform(lambda x: x.max()) == freq_conversion.Counts
    freq_conversion = freq_conversion[idx]
    keys = freq_conversion[codes_precision[0]]
    values = freq_conversion[codes_precision[1]]
    return {k: v for k, v in zip(keys, values)}


def get_data(country1: [str], country2: [str], commodity: [str], flow=(1,), frequency: str = "A",
             year: [str] = ("all",),
             month: [str] = ("",),
             reporting_code: str = "S1",
             proxy: {} = None,
             n_tries=0):
    commodity_ = "".join([str(c) + "," for c in commodity if c != ""])
    commodity_ = commodity_[:-1]
    flow_ = "".join([str(f) for f in flow])
    dates = []
    country1_ = "".join([str(country) + "," for country in country1 if country != ""])[:-1]
    country2_ = "".join([str(country) + "," for country in country2 if country != ""])[:-1]
    if month[0] != "" or year[0] != "all":
        for z, (i, j) in enumerate(product(year, month)):
            if z != len(year) * len(month):
                date = "".join([i, j, ","])[:-1]
            else:
                date = "".join([i, j])
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
    data = "".join([f"{key}={value}&" for key, value in zip(values.keys(), values.values())])[:-1]
    # data = urllib.parse.urlencode(data)
    req = urllib.request.Request(COMTRADE_URL + data, None, headers={"User-Agent": "Chrome/35.0.1916.47"})
    if proxy is not None:
        proxy_ = urllib.request.ProxyHandler(proxy)
        opener = urllib.request.build_opener(proxy_)
        try:
            print("Trying " + COMTRADE_URL + data + " + " + str(proxy))
            with opener.open(req, timeout=120) as response:
                data_response = response.read()
        except Exception as e:
            print(f"Download not successful because of {e} for proxy {proxy}")
            return [{"Error": 1}]
    else:
        print("Trying " + COMTRADE_URL + data)
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                data_response = response.read()
                eventlet.sleep(2)
        except Exception as e:
            # print(f"Tried without proxy after {n_tries} and got error {e}")
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
        if r['ip']:
            list1.append({f"{r['protocols'][0]}": f"{r['ip']}:{r['port']}"})

    list2 = []
    req = urllib.request.Request(URL, None, headers={"User-Agent": "Chrome/35.0.1916.47 "})
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
        if len(set(l[-2].split("-"))) == 3:
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
            # print(proxy)
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
    random_idx = np.random.choice(np.arange(0, len(proxies)), size=len(proxies), replace=False)
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
        assignments = assignments[:len(codes)]
        assigned_proxies = [proxies[i] for i in assignments]
    else:
        assigned_proxies = proxies[:len(codes)]

    p = ["0"] * len(codes)
    cc = ["TOTAL"] * len(codes)
    flow = [1] * len(codes)
    freq = ["A"] * len(codes)
    yrs = ["all"] * len(codes)
    months = [""] * len(codes)
    rc = ["S1"] * len(codes)
    params = [(codes[i], [p[i]], [cc[i]], [flow[i]], freq[i], [yrs[i]], [months[i]], rc[i], assigned_proxies[i])
              for i in range(0, len(codes))]

    batches = len(codes) // len(proxies) + 1
    batch_size = len(proxies)
    for t in range(0, batches):
        print(f"Batch {t}")
        if t < batches:
            batch_params = params[t * batch_size:(t + 1) * batch_size]
        else:
            batch_params = params[t * batch_size:]
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
            print(c, len(y), df[df["Country Code"] == int(c)]["Country Name, Full "].values[0])


def link_func(gt, edge_list, param_idx, q, seen: set, data=None, q_proxy: eventlet.Queue = None, proxy=None):
    if gt is not None:
        data = gt.wait()
    elif data is None:
        raise Exception("Data cannot be None if no gt is available")
    row = {}
    error = False
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

    if bool(row) and not error:
        with open(COMTRADE_DATASET, "wb") as file:
            pkl.dump(edge_list, file)
        with open(SEEN_PARAMS, "wb") as file:
            pkl.dump(seen, file)
            print(f"Q-size {q.qsize()}")
    elif not bool(row) and not error:
        # print("Data empty")
        pass


def build_dataset(flow=("1",), frequency="A", reporting_code="SITC1", use_proxy=True):
    if not os.path.exists(PARAMS_LIST):
        if os.path.exists(DATA_AVAILABILITY_PATH):
            with open(DATA_AVAILABILITY_PATH, "rb") as file:
                availability = pkl.load(file)
        else:
            # check_years_data_availability()
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
        conversion_table = pd.read_excel(CODES_CONVERSION_PATH, index_col=None,
                                         converters={0: str, 1: str, 2: str, 3: str, 4: str, 5: str, 6: str, 7: str,
                                                     8: str, 9: str, 10: str})
        s1 = list(set(conversion_table[reporting_code].dropna().astype(str).str[:2].to_list()))
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
        for y in df_avail.keys():
            len_c5 = len(df_avail[y])
            for c5 in range(0, len_c5):
                countries = tuple(df_avail[y][c5])
                len_p = len(s1)
                for p in range(0, len_p):
                    products = tuple(s1[p])
                    params.append([countries, ("all",), products, tuple(flow), frequency, (y,), ("",), "S1"])
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
    else:
        seen = set()
        params_idx = np.arange(0, len(params))

    if not os.path.exists(COMTRADE_DATASET):
        open(COMTRADE_DATASET, 'a').close()
    if os.path.getsize(COMTRADE_DATASET) > 0:
        with open(COMTRADE_DATASET, "rb") as file:
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

    use_my = False
    time_to_check = time.time()
    time_to_check_proxy = time.time()

    while True:
        while not q.empty():
            if use_proxy:
                t = time.time()
                if t > time_to_check_proxy:
                    proxies = scrape_proxy()
                    proxies = parallelize_check_proxy(proxies)
                    for proxy in proxies:
                        q_proxy.put(proxy)
                    time_to_check = 60 * 60 + time.time()
                # batch_size = len(proxies)
                # batches = q.qsize() // batch_size + 1
                batch_counter = 0
                while not q_proxy.empty():
                    proxy = q_proxy.get()
                    idx = q.get()
                    print("Spawning proxy")
                    thread_param = (*params[idx], proxy)
                    thread = pool.spawn(get_data, *thread_param)
                    thread.link(link_func, edge_list, idx, q, seen, q_proxy=q_proxy, proxy=proxy)
                    batch_counter += 1
                    t = time.time()
                    if t > time_to_check and use_my is False:
                        for i in range(100):
                            try:
                                print("Spawning without proxy")
                                idx = q.get()
                                thread_param = (*params[idx], None)
                                thread = pool.spawn(get_data, *thread_param)
                                result = thread.wait()

                            except urllib.error.HTTPError as e:
                                if e.code == 409:
                                    # print(f"Cannot Use My Machine because of Exception {e}")
                                    time_to_check = 60 * 60 + time.time()
                                    q.put(idx)
                                    use_my = False
                            except Exception as e:
                                # print(f"Trying with my machine Exception {e}")
                                q.put(idx)
                            else:
                                batch_counter += 1
                                use_my = True
                                link_func(None, edge_list=edge_list, param_idx=idx, q=q, seen=seen, data=result)
                    spawn_threads = threads if use_my is False else threads + 100
                    if not batch_counter % spawn_threads:
                        pool.waitall()


if __name__ == "__main__":
    parser = arg.ArgumentParser()
    parser.add_argument("-month", action="store_true", help="Monthly or Annual, default = Annual")
    parser.add_argument("--flow", type=int, default=1, help="1 for import, 2 for export")
    parser.add_argument("--proxy", action="store_true", help="Use proxy")
    args = parser.parse_args()
    monthly = args.month
    proxy = args.proxy
    flow = args.flow
    flow = [str(flow)]

    if monthly:
        save_path = MONTHLY_DATA_PATH
        frequency = "M"
    else:
        save_path = ANNUAL_DATA_PATH
        frequency = "A"
    # load_countries_codes()
    build_dataset(flow=flow, use_proxy=proxy)
