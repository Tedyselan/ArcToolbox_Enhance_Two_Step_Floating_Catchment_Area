# -*- coding: utf-8 -*-
"""
E2SFCA - Python toolbox untuk analisis aksesibilitas fasilitas dengan
metode Enhance 2-Step Floating Catchment Area (E2SFCA)
"""

import os
import numpy as np
import pandas as pd
import arcpy


# ================================================================================
# E2SFCA CORE
# ================================================================================

# Bobot Zona Peluruhan Jarak: Lambat dan Cepat
# ZONE_WEIGHTS_SLOW = [1.00, 0.68, 0.22]
# ZONE_WEIGHTS_SHARP = [1.00, 0.42, 0.09]

METHODS = ("GAUSSIAN", "BINARY", "POWER", "EXPONENTIAL", "ZONES")


def _gaussian(d, d0):
    """Bobot Gaussian ternormalisasi: W(0) = 1, W(d0) = 0."""
    d = np.asarray(d, dtype=float)
    return (np.exp(-0.5 * (d / d0) ** 2) - np.exp(-0.5)) / (1.0 - np.exp(-0.5))


def parse_zone_breaks(text):
    """'2;5;10' -> [2.0, 5.0, 10.0]  (batas atas tiap zona; yang terakhir = d0)"""
    if text is None or str(text).strip() == "":
        raise ValueError("Isi batas atas zona, dipisah titik koma, mis. 2;5;10.")
    try:
        vals = [float(x) for x in str(text).replace(",", ";").split(";") if x.strip()]
    except ValueError:
        raise ValueError("Format batas zona: angka dipisah titik koma, mis. 2;5;10.")
    if len(vals) < 2:
        raise ValueError("Minimal 2 zona (2 nilai batas atas).")
    if vals[0] <= 0:
        raise ValueError("Batas zona harus > 0.")
    if any(b2 <= b1 for b1, b2 in zip(vals, vals[1:])):
        raise ValueError("Batas zona harus naik (nilai berikutnya lebih besar). "
                         "Isi dengan jarak, bukan bobot.")
    return vals


def zone_weights_from_breaks(breaks):
    """Bobot tiap zona = Gaussian ternormalisasi (skala d0 = batas terakhir)
    Zona 1 mulai dari jarak 0."""
    b = np.asarray(breaks, dtype=float)
    lo = np.concatenate(([0.0], b[:-1]))
    return _gaussian((lo + b) / 2.0, b[-1])


def distance_weight(d, d0, method="GAUSSIAN", beta=1.0, zone_breaks=None):
    """
    Bobot distance decay W(d); 0 untuk d > d0.
    GAUSSIAN    : (exp(-0.5 (d/d0)^2) - exp(-0.5)) / (1 - exp(-0.5))
    BINARY      : 1 di dalam catchment (= 2SFCA standar / baseline)
    POWER       : (max(d, 0.05*d0) / (0.05*d0)) ** -beta   (maks 1)
    EXPONENTIAL : exp(-beta * d / d0)
    ZONES       : zona dari batas atas yang diberikan (batas terakhir = d0);
                  bobot tiap zona = Gaussian di titik tengah zona (konstan dalam zona)
    """
    if d0 is None or d0 <= 0:
        raise ValueError("d0 harus > 0.")
    d = np.asarray(d, dtype=float)
    m = str(method).upper()
    if m == "GAUSSIAN":
        w = _gaussian(d, d0)
    elif m == "BINARY":
        w = np.ones_like(d)
    elif m == "POWER":
        floor = 0.05 * d0
        w = (np.maximum(d, floor) / floor) ** (-float(beta))
    elif m == "EXPONENTIAL":
        w = np.exp(-float(beta) * d / d0)
    elif m == "ZONES":
        if zone_breaks is None:
            raise ValueError("Metode ZONES membutuhkan batas atas zona.")
        b = np.asarray(zone_breaks, dtype=float)
        if abs(b[-1] - d0) > 1e-9 * max(1.0, d0):
            raise ValueError("Batas zona terakhir (%s) harus sama dengan d0 (%s)." % (b[-1], d0))
        zw = zone_weights_from_breaks(b)
        # jarak tepat di batas zona masuk zona yang lebih jauh; d = d0 masuk zona terakhir
        idx = np.clip(np.searchsorted(b, d, side="right"), 0, len(b) - 1)
        w = zw[idx]
    else:
        raise ValueError("Metode tidak dikenal: %s (pilih: %s)" % (method, METHODS))
    return np.where(d <= d0, w, 0.0)


def compute_e2sfca(od, facility, demand, d0_step1, d0_step2=None,
                   method="GAUSSIAN", beta=1.0, zone_breaks=None):
    """
    od          : DataFrame [facility_id, demand_id, dist]
    facility    : DataFrame [facility_id, S]
    demand      : DataFrame [demand_id, P]
    return      : off_res [facility_id, S, POP_W, R, N_DM]
                  dem_res [demand_id, P, A, N_FC]   (A = NaN jika P <= 0)

    dengan:
    S       = kapasitas layanan fasilitas
    P       = populasi titik demand
    POP_W   = populasi terbobot (Σ P·W(d))
    R       = ratio fasilitas terhadap populasi (Step-1)
    N_DM    = Jumlah titik demand dalam radius d0 (hitungan) dari fasilitas
    A       = indeks aksesibilitas (Step-2)
    N_FC    = jumlah titik fasilitas dalam radius d0 (hitungan) dari demand
    """

    if d0_step2 is None:
        d0_step2 = d0_step1
 
    facility = facility.copy()
    demand = demand.copy()
    facility["S"] = pd.to_numeric(facility["S"], errors="coerce").fillna(1.0)
    demand["P"] = pd.to_numeric(demand["P"], errors="coerce").fillna(0.0)
 
    od = od[["facility_id", "demand_id", "dist"]].copy()
    od = od.merge(demand[["demand_id", "P"]], on="demand_id", how="inner")
    od = od.merge(facility[["facility_id", "S"]], on="facility_id", how="inner")
 
    # Step 1: R_j = S_j / sum_k P_k W(d_kj), d_kj <= d0_step1
    w1 = distance_weight(od["dist"].values, d0_step1, method, beta, zone_breaks)
    od["pw"] = od["P"].values * w1
    in1 = od["dist"].values <= d0_step1
    pop_w = od.groupby("facility_id")["pw"].sum()
    n_dm = od[in1].groupby("facility_id")["demand_id"].nunique()
 
    off = facility.set_index("facility_id")
    off["POP_W"] = pop_w.reindex(off.index).fillna(0.0)
    off["N_DM"] = n_dm.reindex(off.index).fillna(0).astype(int)
    off["R"] = np.where(off["POP_W"] > 0,
                        off["S"] / off["POP_W"].where(off["POP_W"] > 0, 1.0), 0.0)
 
    # Step 2: A_i = sum_j R_j W(d_ij), d_ij <= d0_step2
    w2 = distance_weight(od["dist"].values, d0_step2, method, beta, zone_breaks)
    od["rw"] = od["facility_id"].map(off["R"]).values * w2
    in2 = od["dist"].values <= d0_step2
    acc = od.groupby("demand_id")["rw"].sum()
    n_fc = od[in2].groupby("demand_id")["facility_id"].nunique()
 
    dem = demand.set_index("demand_id")
    dem["A"] = acc.reindex(dem.index).fillna(0.0)
    dem["N_FC"] = n_fc.reindex(dem.index).fillna(0).astype(int)
    dem.loc[dem["P"] <= 0, "A"] = np.nan
    return off.reset_index(), dem.reset_index()


# ================================================================================
# HELPER ARCPY
# ================================================================================

def _msg(text):
    arcpy.AddMessage(text)
 
 
def _warn(text):
    arcpy.AddWarning(text)
 
 
def _same_sr(a, b):
    """True jika dua spatial reference sama (kode EPSG/WKID, atau nama jika kode tidak ada)."""
    ca, cb = getattr(a, "factoryCode", 0), getattr(b, "factoryCode", 0)
    if ca and cb:
        return ca == cb
    return getattr(a, "name", None) == getattr(b, "name", None)


def _network_sr(network):
    """Spatial reference network dataset"""
    d = arcpy.Describe(network)
    if hasattr(d, "dataElement"):
        d = d.dataElement
    sr = getattr(d, "spatialReference", None)
    if sr is None:
        raise ValueError("CRS network dataset tidak dapat dibaca.")
    return sr


def _crs_errors(net_sr, fac_sr, dm_sr):
    """
    Pelanggaran CRS yang menghentikan proses: daftar (objek, pesan).
    Syarat: ketiga data harus punya CRS, dan network dataset berCRS proyeksi.
    """
    errs = []
    for name, sr in (("network dataset", net_sr), ("facility", fac_sr), ("demand", dm_sr)):
        if getattr(sr, "name", "Unknown") in ("", "Unknown", None):
            errs.append((name, "%s tidak punya sistem koordinat (Unknown). "
                         "Definisikan CRS-nya terlebih dahulu." % name.capitalize()))
    if errs:
        return errs
    if net_sr.type != "Projected":
        return [("network dataset",
                 "CRS network dataset (%s) bukan sistem koordinat proyeksi. Network harus "
                 "berCRS proyeksi agar facility dan demand dapat disamakan ke CRS "
                 "proyeksi yang sama." % net_sr.name)]
    return []
 
 
def _assert_crs(fc, name, net_sr):
    """Pastikan layer hasil persiapan benar-benar berada di CRS network."""
    sr = arcpy.Describe(fc).spatialReference
    if not _same_sr(sr, net_sr):
        raise RuntimeError("Kontrol CRS gagal: layer %s hasil persiapan berada di %s, "
                           "bukan di CRS network (%s)." % (name, getattr(sr, "name", "?"), net_sr.name))
 
 
def _num(p, default=None):
    """Nilai numerik parameter opsional."""
    v = p.value
    return default if v in (None, "") else float(v)
 
 
def _by_name(parameters):
    return {x.name: x for x in parameters}
 
 
def _read_table(src, fields, names):
    return pd.DataFrame([tuple(r) for r in arcpy.da.SearchCursor(src, fields)],
                        columns=names)
 
 
def _delete_if_exists(path):
    if arcpy.Exists(path):
        arcpy.management.Delete(path)


# ================================================================================
# LOGICAL MECHANISM FOR EACH TOOL
# ================================================================================

# -- 1. Prepare Data
def _fix_centroids_outside(poly_fc, pt_fc):
    """
    Centroid yang jatuh di luar poligonnya (poligon cekung, pesisir, kepulauan)
    dipindahkan ke label point yang dijamin di dalam poligon.
    
    Mengembalikan jumlahnya.
    """
    pts = {r[0]: r[1] for r in arcpy.da.SearchCursor(pt_fc, ["ORIG_FID", "SHAPE@"])}
    fix = {}
    for oid, poly in arcpy.da.SearchCursor(poly_fc, ["OID@", "SHAPE@"]):
        pt = pts.get(oid)
        if poly is None or pt is None:
            continue
        if not poly.contains(pt):
            fix[oid] = arcpy.PointGeometry(poly.labelPoint, poly.spatialReference)
    if fix:
        with arcpy.da.UpdateCursor(pt_fc, ["ORIG_FID", "SHAPE@"]) as cur:
            for orig, _ in cur:
                if orig in fix:
                    cur.updateRow([orig, fix[orig]])
    return len(fix)


def _datum_name(sr):
    """Nama datum spatial reference (proyeksi atau geografis); None jika tak terbaca."""
    datum = getattr(sr, "datumName", None)
    if not datum:
        datum = getattr(getattr(sr, "GCS", None), "datumName", None)
    return datum or None


def _pick_transformation(src_sr, target_sr, extent=None):
    """Pilih transformasi datum untuk arcpy.management.Project"""
    d1, d2 = _datum_name(src_sr), _datum_name(target_sr)
    if not d1 or not d2 or d1 == d2:
        return "", None, False
    try:
        cands = (arcpy.ListTransformations(src_sr, target_sr, extent) if extent is not None
                 else arcpy.ListTransformations(src_sr, target_sr))
    except Exception:
        cands = []
    if cands:
        msg = ("  Datum berbeda (%s -> %s); transformasi dipakai: %s%s" % (
            d1, d2, cands[0],
            " (dari %d kandidat)" % len(cands) if len(cands) > 1 else ""))
        return cands[0], msg, False
    return "", ("Datum berbeda (%s -> %s) tetapi tidak ada transformasi datum yang "
                "ditemukan; layer diproyeksikan tanpa transformasi sehingga posisi bisa "
                "bergeser. Samakan datum data sebelum menjalankan tool." % (d1, d2)), True


def _clean_layer(src, out, target_sr, as_centroid, tag):
    """
    Salin layer ke output.
    
    as_centroid=True: poligon -> titik centroid.
    """
    work, tmp = src, None
    if target_sr:
        tmp = "memory\\_e2_tmp_" + tag
        _delete_if_exists(tmp)
        d = arcpy.Describe(src)
        xform, note, is_warn = _pick_transformation(
            d.spatialReference, target_sr, getattr(d, "extent", None))
        if note:
            (_warn if is_warn else _msg)(note)
        if xform:
            arcpy.management.Project(src, tmp, target_sr, xform)
        else:
            arcpy.management.Project(src, tmp, target_sr)
        work = tmp
    if as_centroid:
        arcpy.management.FeatureToPoint(work, out, "CENTROID")
        n_fix = _fix_centroids_outside(work, out)
        if n_fix:
            _msg("  %d centroid jatuh di luar poligonnya; dipindahkan ke titik di "
                 "dalam poligon." % n_fix)
    else:
        arcpy.management.CopyFeatures(work, out)
    if tmp:
        arcpy.management.Delete(tmp)
 
 
def _rewrite_fields(fc, src_fields, new_specs, builder):
    """Baca nilai sumber -> hapus field bentrok -> tambah field baru -> tulis."""
    rows = [list(r) for r in arcpy.da.SearchCursor(fc, ["OID@"] + src_fields)]
    existing = {f.name.upper() for f in arcpy.ListFields(fc)}
    for name, _, _ in new_specs:
        if name.upper() in existing:
            arcpy.management.DeleteField(fc, name)
    for name, ftype, flen in new_specs:
        if ftype == "TEXT":
            arcpy.management.AddField(fc, name, "TEXT", field_length=flen)
        else:
            arcpy.management.AddField(fc, name, ftype)
    new_names = [n for n, _, _ in new_specs]
    values = {r[0]: builder(r[0], r[1:]) for r in rows}
    with arcpy.da.UpdateCursor(fc, ["OID@"] + new_names) as cur:
        for r in cur:
            cur.updateRow([r[0]] + list(values[r[0]]))
    return values
 
 
def run_prepare(facility_fc, facility_id_f, capacity_f, demand_fc, demand_id_f,
                pop_f, network, out_facility, out_demand):
    # CRS: kedua layer disiapkan (sebagai data sementara) di CRS network dataset, yang
    # dipakai untuk menghitung jarak. Layer input asli tidak diubah; layer yang CRS-nya
    # sudah sama dengan network disalin tanpa proyeksi.
    net_sr = _network_sr(network)
    fac_sr = arcpy.Describe(facility_fc).spatialReference
    dm_sr = arcpy.Describe(demand_fc).spatialReference
    errs = _crs_errors(net_sr, fac_sr, dm_sr)
    if errs:
        raise ValueError(errs[0][1])
    # Layer yang CRS-nya berbeda dari network dataset ditransformasikan ke CRS network
    # dataset dan disimpan sebagai data sementara (fac / dm).
    fac_target = None if _same_sr(fac_sr, net_sr) else net_sr
    dm_target = None if _same_sr(dm_sr, net_sr) else net_sr
    for name, sr, tgt in (("facility", fac_sr, fac_target), ("demand", dm_sr, dm_target)):
        if tgt:
            _msg("CRS %s (%s) berbeda dengan network (%s): diproyeksikan ke CRS network "
                 "dan disimpan sebagai data sementara." % (name, sr.name, net_sr.name))
 
    # --- facility
    _msg("Menyiapkan data fasilitas ...")
    _clean_layer(facility_fc, out_facility, fac_target, False, "fac")
    _assert_crs(out_facility, "fasilitas", net_sr)
    src = [f for f in (facility_id_f, capacity_f) if f]
    stats = {"cap_fix": 0}
 
    def build_facility(oid, row):
        i = 0
        if facility_id_f:
            pid = row[i]
            i += 1
            pid = None if pid is None else str(pid).strip()
        else:
            pid = str(oid)
        s = 1.0
        if capacity_f:
            v = row[i]
            if v is None or v <= 0:
                stats["cap_fix"] += 1
            else:
                s = float(v)
        return (pid, s)
 
    vals = _rewrite_fields(out_facility, src,
                           [("FC_ID", "TEXT", 50), ("FC_S", "DOUBLE", None)],
                           build_facility)
    ids = [v[0] for v in vals.values()]
    if any(i in (None, "") for i in ids):
        raise ValueError("ID fasilitas ada yang kosong.")
    if len(set(ids)) != len(ids):
        raise ValueError("ID fasilitas tidak unik.")
    _msg("  %d fasilitas; %d nilai kapasitas kosong/<=0 diganti 1%s" % (
        len(ids), stats["cap_fix"],
        "" if capacity_f else " (kolom kapasitas tidak diisi: semua S = 1)"))
 
    # --- demand
    shape = arcpy.Describe(demand_fc).shapeType     # "Polygon" / "Point" / ...
    if shape == "Polygon":
        _msg("Input demand berupa poligon: menghitung centroid ...")
        as_centroid = True
    elif shape == "Point":
        _msg("Input demand berupa titik: perhitungan centroid dilewati.")
        as_centroid = False
    else:
        raise ValueError("Input demand harus poligon atau titik (terdeteksi: %s)." % shape)
    _clean_layer(demand_fc, out_demand, dm_target, as_centroid, "dem")
    _assert_crs(out_demand, "demand", net_sr)
    stats2 = {"pop_fix": 0, "total": 0.0}
 
    def build_demand(oid, row):
        did, p = row
        did = None if did is None else str(did).strip()
        if p is None or p < 0:
            stats2["pop_fix"] += 1
            p = 0.0
        stats2["total"] += float(p)
        return (did, float(p))
 
    vals = _rewrite_fields(out_demand, [demand_id_f, pop_f],
                           [("DM_ID", "TEXT", 50), ("DM_POP", "DOUBLE", None)],
                           build_demand)
    ids = [v[0] for v in vals.values()]
    if any(i in (None, "") for i in ids):
        raise ValueError("ID poligon demand ada yang kosong.")
    if len(set(ids)) != len(ids):
        raise ValueError("ID poligon demand tidak unik.")
    _msg("  %d titik demand; total populasi %.0f; %d populasi null/negatif dianggap 0"
         % (len(ids), stats2["total"], stats2["pop_fix"]))
    _msg("Selesai.")


# -- 2. OD Cost Matrix
_TO_M = {"meters": 1.0, "kilometers": 1000.0, "miles": 1609.344, "feet": 0.3048,
         "yards": 0.9144, "nauticalmiles": 1852.0, "decimeters": 0.1,
         "centimeters": 0.01, "millimeters": 0.001}
 
 
def _impedance_units(network, impedance_name):
    try:
        d = arcpy.Describe(network)
        if hasattr(d, "dataElement"):
            d = d.dataElement
        for a in d.attributes:
            if a.name == impedance_name:
                return a.units
    except Exception:
        pass
    return None


_MODE_CACHE = {}   # {path network: [nama travel mode]}; updateParameters jalan berulang

def _network_key(param):
    """Cache key untuk parameter network: path data sources"""
    src = getattr(param.value, "dataSource", None)
    return str(src) if src else param.valueAsText


def _travel_modes(network):
    if network not in _MODE_CACHE:
        _MODE_CACHE[network] = list(arcpy.nax.GetTravelModes(network).keys())
    return _MODE_CACHE[network]


_NAX_UNIT = {"meters": "Meters", "kilometers": "Kilometers", "miles": "Miles",
             "feet": "Feet", "yards": "Yards", "nauticalmiles": "NauticalMiles",
             "decimeters": "Decimeters", "centimeters": "Centimeters",
             "millimeters": "Millimeters"}


def _od_unit_plan(imp_units, dist_unit, d0, travel_mode="", impedance=""):
    """Samakan satuan solver dengan satuan atribut impedansi agar cutoff tidak ambigu
    (dibaca dalam satuan impedansi ATAU distanceUnits, hasilnya sama)."""

    if not imp_units:
        raise ValueError("Satuan impedansi travel mode '%s' (%s) tidak terbaca dari "
                         "network dataset; tidak bisa memastikan satuan cutoff."
                         % (travel_mode, impedance))
    key = imp_units.replace(" ", "").lower()
    if key not in _TO_M or key not in _NAX_UNIT:
        raise ValueError("Impedansi travel mode '%s' (%s) bukan atribut jarak "
                         "(satuan: %s). Pilih travel mode berimpedansi panjang jalan."
                         % (travel_mode, impedance, imp_units))
    u = dist_unit.lower()
    d0_imp = float(d0) * _TO_M[u] / _TO_M[key]
    back = _TO_M[key] / _TO_M[u]
    return key, d0_imp, back
 
 
def _pick_cost_field(fields, unit, impedance_name):
    totals = [f for f in fields if f.lower().startswith("total_")]
    for f in totals:
        if f.lower().endswith("_" + unit.lower()):
            return f
    for f in totals:
        if impedance_name.lower() in f.lower():
            return f
    if len(totals) == 1:
        return totals[0]
    raise RuntimeError("Tidak bisa menentukan kolom jarak pada output OD. "
                       "Kolom 'Total_*' yang ada: %s" % totals)


def _read_status(fc):
    """{Name: Status} dari output Origins/Destinations solver; None jika tak ada field Status."""
    if "Status" not in {f.name for f in arcpy.ListFields(fc)}:
        return None
    return {str(r[0]): r[1] for r in arcpy.da.SearchCursor(fc, ["Name", "Status"])}
 
 
def _report_unlocated(status, label, pop_by=None, show=10):
    """
    Laporkan titik yang tidak bisa dipakai solver (Status <> 0, artinya tidak OK:
    tidak terlokasi di jaringan, elemen tak dapat dilalui, dst.).
    Mengembalikan jumlah titik berstatus OK, atau None jika Status tidak tersedia.
    """
    if status is None:
        return None
    bad = sorted(k for k, s in status.items() if s not in (0, None))
    if bad:
        txt = ", ".join(bad[:show]) + (" ..." if len(bad) > show else "")
        extra = ""
        if pop_by:
            extra = " (populasi %.0f)" % sum(pop_by.get(k, 0.0) for k in bad)
        codes = sorted({status[k] for k in bad})
        _warn("%d %s tidak dapat dipakai solver%s, Status solver = %s (0 = OK): %s. "
              "Biasanya titik berada di luar jangkauan jaringan atau terlalu jauh dari "
              "ruas jalan (search tolerance); periksa cakupan jaringan dan CRS."
              % (len(bad), label, extra, codes, txt))
    return len(status) - len(bad)

 
def _no_pairs_message(ok_o, ok_d, d0, dist_unit):
    """Penjelasan jika tabel OD kosong, sesuai jumlah titik yang terlokasi."""
    if ok_o == 0 or ok_d == 0:
        why = ("Tidak ada %s yang terlokasi di jaringan." % (
            "titik fasilitas dan titik demand" if ok_o == 0 and ok_d == 0
            else "titik fasilitas" if ok_o == 0 else "titik demand"))
        fix = ("Periksa apakah data berada di dalam cakupan network dataset, CRS, dan "
               "jarak titik ke ruas jalan.")
    elif ok_o is not None and ok_d is not None:
        why = ("%d titik fasilitas dan %d titik demand terlokasi, tetapi tidak ada pasangan "
               "dengan jarak jaringan <= d0 (%s %s)." % (ok_o, ok_d, d0, dist_unit))
        fix = ("Naikkan d0, periksa satuan jarak, atau periksa konektivitas jaringan "
               "(titik bisa berada di komponen jaringan yang terpisah).")
    else:
        why = ("Tidak ada pasangan fasilitas-demand dengan jarak jaringan <= d0 (%s %s). "
               "Penyebab yang mungkin: d0 terlalu kecil, titik tidak terlokasi di jaringan, "
               "atau jaringan tidak terhubung." % (d0, dist_unit))
        fix = "Periksa pesan [solver] di atas, d0, satuan jarak, CRS, dan cakupan jaringan."
    return "Tabel OD kosong. %s %s" % (why, fix)
 
 
def run_od(facility_fc, demand_fc, network, travel_mode, d0_max, dist_unit, out_table):
    if arcpy.CheckExtension("Network") != "Available":
        raise RuntimeError("Lisensi Network Analyst tidak tersedia.")
    arcpy.CheckOutExtension("Network")
    tmp_paths = []
    try:
        modes = arcpy.nax.GetTravelModes(network)
        if travel_mode not in modes:
            raise ValueError("Travel mode '%s' tidak ada. Pilihan: %s" % (
                travel_mode, ", ".join(modes.keys())))
        tm = modes[travel_mode]
 
        units = _impedance_units(network, tm.impedance)
        ukey, d0_imp, back = _od_unit_plan(units, dist_unit, d0_max,
                                           travel_mode, tm.impedance)
        unit_enum = getattr(arcpy.nax.DistanceUnits, _NAX_UNIT[ukey])
        _msg("Satuan impedansi network: %s; d0 %s %s dikirim ke solver sebagai %s %s"
             % (units, d0_max, dist_unit, d0_imp, units))
 
        odcm = arcpy.nax.OriginDestinationCostMatrix(network)
        odcm.travelMode = tm
        odcm.distanceUnits = unit_enum
        odcm.defaultImpedanceCutoff = d0_imp
        odcm.defaultDestinationCount = None
        odcm.lineShapeType = arcpy.nax.LineShapeType.NoLine
 
        in_o = arcpy.nax.OriginDestinationCostMatrixInputDataType.Origins
        in_d = arcpy.nax.OriginDestinationCostMatrixInputDataType.Destinations
        fm = odcm.fieldMappings(in_o, True)
        fm["Name"].mappedFieldName = "FC_ID"
        odcm.load(in_o, facility_fc, fm)
        fm = odcm.fieldMappings(in_d, True)
        fm["Name"].mappedFieldName = "DM_ID"
        odcm.load(in_d, demand_fc, fm)
 
        _msg("Menghitung OD Cost Matrix (cutoff %s %s) ..." % (d0_max, dist_unit))
        result = odcm.solve()
        for _, txt in result.solverMessages(arcpy.nax.MessageSeverity.All):
            _msg("  [solver] %s" % txt)
        if not result.solveSucceeded:
            raise RuntimeError("Solver gagal. Lihat pesan [solver] di atas.")
 
        out_t = arcpy.nax.OriginDestinationCostMatrixOutputDataType
        lines, origins, dests = ("memory\\_e2_od_lines", "memory\\_e2_od_origins",
                                 "memory\\_e2_od_dests")
        tmp_paths = [lines, origins, dests]
        result.export(out_t.Lines, lines)
        result.export(out_t.Origins, origins)
        result.export(out_t.Destinations, dests)
 
        cost_f = _pick_cost_field([f.name for f in arcpy.ListFields(lines)],
                                  ukey, tm.impedance)
        _msg("  Kolom jarak: %s" % cost_f)
        oname = {r[0]: r[1] for r in arcpy.da.SearchCursor(origins, ["OID@", "Name"])}
        dname = {r[0]: r[1] for r in arcpy.da.SearchCursor(dests, ["OID@", "Name"])}
 
        folder, name = os.path.split(str(out_table))
        arcpy.management.CreateTable(folder, name)
        arcpy.management.AddField(out_table, "FC_ID", "TEXT", field_length=50)
        arcpy.management.AddField(out_table, "DM_ID", "TEXT", field_length=50)
        arcpy.management.AddField(out_table, "DIST", "DOUBLE")
 
        n, dmax = 0, 0.0
        hit_o, hit_d = set(), set()
        with arcpy.da.InsertCursor(out_table, ["FC_ID", "DM_ID", "DIST"]) as ins:
            for o, d, c in arcpy.da.SearchCursor(
                    lines, ["OriginOID", "DestinationOID", cost_f]):
                if c is None:
                    continue
                c = float(c) * back
                ins.insertRow((oname[o], dname[d], c))
                hit_o.add(oname[o])
                hit_d.add(dname[d])
                n += 1
                dmax = max(dmax, float(c))
        _msg("  %d pasangan OD; jarak maksimum %.3f %s" % (n, dmax, dist_unit))
        _msg("  Fasilitas terhubung: %d, titik demand terhubung: %d"
             % (len(hit_o), len(hit_d)))

        # titik tidak terjangkau jaringan
        pop_by = {str(r[0]): (r[1] or 0.0)
                  for r in arcpy.da.SearchCursor(demand_fc, ["DM_ID", "DM_POP"])}
        ok_o = _report_unlocated(_read_status(origins), "titik fasilitas")
        ok_d = _report_unlocated(_read_status(dests), "titik demand", pop_by)
        if ok_o is None or ok_d is None:
            _msg("  Field Status tidak ada pada output solver; titik yang tidak terlokasi "
                 "tidak dapat dipisahkan dari titik di luar cutoff.")
        else:
            _msg("  Terlokasi di jaringan: %d titik fasilitas, %d titik demand" % (ok_o, ok_d))
            lost_o, lost_d = ok_o - len(hit_o), ok_d - len(hit_d)
            if lost_o > 0:
                _msg("  %d titik fasilitas terlokasi tetapi tidak punya titik demand dalam cutoff."
                     % lost_o)
            if lost_d > 0:
                _warn("%d titik demand terlokasi tetapi tidak punya titik fasilitas dalam "
                      "cutoff %s %s (akses = 0)." % (lost_d, d0_max, dist_unit))
        if n == 0:
            raise RuntimeError(_no_pairs_message(ok_o, ok_d, d0_max, dist_unit))
        if dmax > float(d0_max) * 1.01:
            _warn("Jarak maksimum melebihi cutoff. Periksa satuan impedansi network "
                  "vs satuan yang dipilih.")
    finally:
        for t in tmp_paths:
            if arcpy.Exists(t):
                arcpy.management.Delete(t)
        arcpy.CheckInExtension("Network")


# -- 3. Calculate E2SFCA
def _write_results(src, out, key_field, df, key_col, spec):
    """Salin src -> out, lalu tulis kolom df. spec: [(field, tipe, kolom_df)]"""
    arcpy.management.CopyFeatures(src, out)
    existing = {f.name.upper() for f in arcpy.ListFields(out)}
    for fname, ftype, _ in spec:
        if fname.upper() in existing:
            arcpy.management.DeleteField(out, fname)
        arcpy.management.AddField(out, fname, ftype)
    lookup = df.set_index(key_col)
    names = [s[0] for s in spec]
    with arcpy.da.UpdateCursor(out, [key_field] + names) as cur:
        for row in cur:
            k = row[0]
            if k in lookup.index:
                vals = []
                for _, ftype, col in spec:
                    v = lookup.at[k, col]
                    if pd.isna(v):
                        vals.append(None)
                    else:
                        vals.append(int(v) if ftype == "LONG" else float(v))
                cur.updateRow([k] + vals)
 
 
def run_calculate(facility_fc, demand_fc, od_table, d0_1, d0_2, method, beta,
                  zone_text, scale, out_demand, out_facility):
    method = (method or "GAUSSIAN").upper()
    d0_2 = d0_1 if d0_2 is None else d0_2
    zb = parse_zone_breaks(zone_text) if method == "ZONES" else None
 
    facility = _read_table(facility_fc, ["FC_ID", "FC_S"], ["facility_id", "S"])
    demand = _read_table(demand_fc, ["DM_ID", "DM_POP"], ["demand_id", "P"])
    od = _read_table(od_table, ["FC_ID", "DM_ID", "DIST"],
                     ["facility_id", "demand_id", "dist"])
    _msg("Input: %d titik fasilitas, %d titik demand, %d pasangan OD" % (
        len(facility), len(demand), len(od)))
 
    off_res, dem_res = compute_e2sfca(od, facility, demand, d0_1, d0_2, method, beta, zb)
    dem_res["A_SCL"] = dem_res["A"] * scale
 
    no_pop = int((off_res["POP_W"] <= 0).sum())
    if no_pop:
        _warn("%d fasilitas tanpa populasi dalam catchment step 1 (R = 0)." % no_pop)
    zero = dem_res[(dem_res["P"] > 0) & (dem_res["A"] == 0)]
    _msg("Titik demand berpopulasi tanpa akses (A = 0): %d (populasi %.0f)" % (
        len(zero), zero["P"].sum()))
 
    _write_results(demand_fc, out_demand, "DM_ID", dem_res, "demand_id",
                   [("A_RAW", "DOUBLE", "A"), ("A_SCL", "DOUBLE", "A_SCL"),
                    ("N_FC", "LONG", "N_FC")])
    _write_results(facility_fc, out_facility, "FC_ID", off_res, "facility_id",
                   [("R_J", "DOUBLE", "R"), ("POP_W", "DOUBLE", "POP_W"),
                    ("N_DM", "LONG", "N_DM")])
    _msg("Selesai. A_SCL = akses x %.0f (fasilitas per %.0f penduduk, untuk S = 1)."
         % (scale, scale))


# -- 4. Classify & Report
def jenks_edges(values, k, max_n=3000):
    """Natural Breaks (Jenks / Fisher): batas atas kelas 1..k-1 yang meminimalkan
    jumlah kuadrat simpangan di dalam kelas."""
    v = np.sort(np.asarray([x for x in values if x is not None], dtype=float))
    if len(v) > max_n:
        v = np.quantile(v, np.linspace(0.0, 1.0, max_n))
    k = min(int(k), len(np.unique(v)))
    if k < 2:
        return np.array([])
    n = len(v)
    c1 = np.concatenate(([0.0], np.cumsum(v)))
    c2 = np.concatenate(([0.0], np.cumsum(v * v)))
    idx = np.arange(n)
    D = np.full((k, n), np.inf)
    B = np.zeros((k, n), dtype=int)
    D[0] = (c2[1:] - c2[0]) - (c1[1:] - c1[0]) ** 2 / (idx + 1)
    for m in range(1, k):
        for i in range(m, n):
            j = np.arange(m, i + 1)                       # awal kelas ke-(m+1)
            s1 = c1[i + 1] - c1[j]
            cost = D[m - 1][j - 1] + (c2[i + 1] - c2[j]) - s1 * s1 / (i - j + 1)
            b = int(np.argmin(cost))
            D[m][i] = cost[b]
            B[m][i] = j[b]
    edges, end = [], n - 1
    for m in range(k - 1, 0, -1):
        start = int(B[m][end])
        edges.append(v[start - 1])
        end = start - 1
    return np.unique(edges)
 
 
def _class_edges(vals, method, n, breaks_text):
    v = np.asarray([x for x in vals if x is not None], dtype=float)
    m = method.upper()
    if m == "QUANTILE":
        return np.unique(np.quantile(v, np.linspace(0, 1, n + 1)[1:-1]))
    if m == "MANUAL":
        return np.array(sorted(float(x) for x in
                               breaks_text.replace(",", ";").split(";") if x.strip()))
    if m == "EQUAL INTERVAL":
        return np.linspace(v.min(), v.max(), n + 1)[1:-1]
    return jenks_edges(v, n)  # NATURAL BREAKS
 
 
def run_report(admin_fc, admin_id_f, result_fc, method, n_classes, breaks_text,
               threshold, out_polygon):
    method = (method or "NATURAL BREAKS").upper()
    if method == "MANUAL" and not breaks_text:
        raise ValueError("Metode MANUAL membutuhkan batas manual.")
 
    res = {str(r[0]): (r[1], r[2], r[3]) for r in arcpy.da.SearchCursor(
        result_fc, ["DM_ID", "DM_POP", "A_SCL", "N_FC"])}
    arcpy.management.CopyFeatures(admin_fc, out_polygon)
 
    existing = {f.name.upper() for f in arcpy.ListFields(out_polygon)}
    spec = [("POP", "DOUBLE"), ("A_SCL", "DOUBLE"), ("N_FC", "LONG"),
            ("A_CLS", "SHORT"), ("UNDERSRV", "SHORT")]
    for f, t in spec:
        if f in existing:
            arcpy.management.DeleteField(out_polygon, f)
        arcpy.management.AddField(out_polygon, f, t)
 
    a_vals = [v[1] for v in res.values() if v[1] is not None]
    if not a_vals:
        raise ValueError("Tidak ada nilai aksesibilitas pada hasil perhitungan " \
                         "E2SFCA.")
    edges = _class_edges(a_vals, method, n_classes, breaks_text)
    _msg("Batas kelas: %s" % np.round(edges, 5).tolist())
 
    tot_pop = under_pop = 0.0
    n_under = n_match = 0
    with arcpy.da.UpdateCursor(out_polygon, [admin_id_f, "POP", "A_SCL", "N_FC",
                                             "A_CLS", "UNDERSRV"]) as cur:
        for row in cur:
            key = None if row[0] is None else str(row[0]).strip()
            if key not in res:
                continue
            p, a, npo = res[key]
            n_match += 1
            cls = und = None
            if a is not None:
                cls = int(np.digitize(a, edges, right=True)) + 1
                und = 1 if a <= threshold else 0
                tot_pop += p or 0
                if und:
                    n_under += 1
                    under_pop += p or 0
            cur.updateRow([row[0], p, a, npo, cls, und])
 
    _msg("Poligon cocok dengan hasil: %d" % n_match)
    _msg("Underserved (A_SCL <= %s): %d wilayah, populasi %.0f dari %.0f (%.1f%%)" % (
        threshold, n_under, under_pop, tot_pop,
        100.0 * under_pop / tot_pop if tot_pop else 0.0))


# ================================================================================
# TOOLBOX & TOOL CLASSES
# ================================================================================

def _gdb_root(path):
    """Folder .gdb yang memuat path (juga jika di dalam feature dataset)."""
    p = str(path)
    while p:
        if p.lower().endswith(".gdb"):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            return None
        p = parent


def _output_target(path):
    """
    Tentukan jenis output.
      ("gdb", folder_gdb, path)   feature class di dalam geodatabase
      ("shp", folder, path.shp)   shapefile di folder yang sudah ada
      None                        tidak valid
    """
    p = str(path)
    gdb = _gdb_root(p)
    if gdb:
        return ("gdb", gdb, p)
    folder = os.path.dirname(p)
    if folder and os.path.isdir(folder):
        if not p.lower().endswith(".shp"):
            p += ".shp"
        return ("shp", folder, p)
    return None


# Shapefile tidak menyimpan NULL, jadi nilai kosong diganti kode khusus
SHP_NULL_A = -9999.0    # A_SCL     : populasi 0, tidak dinilai
SHP_NULL_CLS = 0        # A_CLS     : tidak dikelaskan (kelas valid mulai dari 1)
SHP_NULL_UND = -1       # UNDERSRV  : tidak dinilai
 
 
def _shp_nulls(a, cls, und):
    return (SHP_NULL_A if a is None else a,
            SHP_NULL_CLS if cls is None else cls,
            SHP_NULL_UND if und is None else und)


def _shp_field_names(names):
    """Peta {nama lama: nama baru} agar semua nama field muat di shapefile (<= 10 karakter)
    dan unik tanpa membedakan huruf besar/kecil."""
    used = {n.upper() for n in names if len(n) <= 10}
    mapping = {}
    for n in names:
        if len(n) <= 10:
            continue
        new = n[:10]
        i = 0
        while new.upper() in used:
            i += 1
            s = str(i)
            new = n[:10 - len(s)] + s
        used.add(new.upper())
        mapping[n] = new
    return mapping

 
def _export_shapefile(src_fc, out_shp):
    """Ekspor hasil ke shapefile; NULL pada A_SCL, A_CLS, UNDERSRV diganti kode khusus."""
    tmp = "memory\\_e2_shp"
    _delete_if_exists(tmp)
    arcpy.management.CopyFeatures(src_fc, tmp)
    skip = {"shape_length", "shape_area"}
    fnames = [f.name for f in arcpy.ListFields(tmp)
              if f.type not in ("OID", "Geometry") and f.name.lower() not in skip]
    ren = _shp_field_names(fnames)
    for old, new in ren.items():
        arcpy.management.AlterField(tmp, old, new)
    if ren:
        _msg("Nama field > 10 karakter dipotong untuk shapefile: %s" % ", ".join(
            "%s -> %s" % (o, n) for o, n in ren.items()))
    n = 0
    with arcpy.da.UpdateCursor(tmp, ["A_SCL", "A_CLS", "UNDERSRV"]) as cur:
        for a, c, u in cur:
            new = _shp_nulls(a, c, u)
            if new != (a, c, u):
                n += 1
                cur.updateRow(list(new))
    arcpy.management.CopyFeatures(tmp, out_shp)
    arcpy.management.Delete(tmp)
    if n:
        _msg("Shapefile tidak menyimpan NULL: %d wilayah tidak dinilai diberi kode "
             "A_SCL = %s, A_CLS = %s, UNDERSRV = %s." % (
                 n, SHP_NULL_A, SHP_NULL_CLS, SHP_NULL_UND))
    return None


class Toolbox:
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "Enhance 2-Step Floating Catchment Area"
        self.alias = "E2SFCA"

        # List of tool classes associated with this toolbox
        self.tools = [E2SFCA]


class E2SFCA:
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "E2SFCA"
        self.description = "Perhitungan aksesibilitas spasial dengan E2SFCA, serta klasifikasi " \
                           "dan penandaan wilayah underserved hasil perhitungan."
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define the tool parameters."""
        params = [
            # -- Facility
            arcpy.Parameter(displayName="Facility Location",
                            name="in_fac_fc",
                            datatype="GPFeatureLayer",
                            parameterType="Required",
                            direction="Input"),
 
            arcpy.Parameter(displayName="Facility - ID Field (Default = ObjectID)",
                            name="fac_id_field",
                            datatype="Field",
                            parameterType="Optional",
                            direction="Input"),
 
            arcpy.Parameter(displayName="Facility - Capacity Field (Default = 1)",
                            name="fac_capacity_field",
                            datatype="Field",
                            parameterType="Optional",
                            direction="Input"),

            # -- Demand
            arcpy.Parameter(displayName="Demand Location",
                            name="in_demand_fc",
                            datatype="GPFeatureLayer",
                            parameterType="Required",
                            direction="Input"),
 
            arcpy.Parameter(displayName="Demand - ID Field",
                            name="demand_id_field",
                            datatype="Field",
                            parameterType="Required",
                            direction="Input"),
 
            arcpy.Parameter(displayName="Demand - Population Field",
                            name="demand_pop_field",
                            datatype="Field",
                            parameterType="Required",
                            direction="Input"),

            # -- Network Dataset
            arcpy.Parameter(displayName="Network Dataset",
                            name="network_dataset",
                            datatype="GPNetworkDatasetLayer",
                            parameterType="Required",
                            direction="Input"),
 
            arcpy.Parameter(displayName="Network Dataset - Travel Mode",
                            name="travel_mode",
                            datatype="GPString",
                            parameterType="Required",
                            direction="Input"),
 
            arcpy.Parameter(displayName="Network Dataset - Distance Unit",
                            name="dist_unit",
                            datatype="GPString",
                            parameterType="Required",
                            direction="Input"),

            # -- E2SFCA
            arcpy.Parameter(displayName="E2SFCA - Distance Threshold (d0)",
                            name="d0",
                            datatype="GPDouble",
                            parameterType="Required",
                            direction="Input"),
 
            arcpy.Parameter(displayName="E2SFCA - Distance Decay Function",
                            name="method",
                            datatype="GPString",
                            parameterType="Required",
                            direction="Input"),
 
            arcpy.Parameter(displayName="E2SFCA - Beta (for POWER / EXPONENTIAL)",
                            name="beta",
                            datatype="GPDouble",
                            parameterType="Optional",
                            direction="Input"),
 
            arcpy.Parameter(displayName="E2SFCA - Zone Break (e.g.: 5; 10; 15)",
                            name="zone_breaks",
                            datatype="GPString",
                            parameterType="Optional",
                            direction="Input"),
 
            arcpy.Parameter(displayName="E2SFCA - Result Scale Factor",
                            name="scale",
                            datatype="GPDouble",
                            parameterType="Required",
                            direction="Input"),

            # -- Klasifikasi
            arcpy.Parameter(displayName="Classification Method",
                            name="class_method",
                            datatype="GPString",
                            parameterType="Required",
                            direction="Input"),
 
            arcpy.Parameter(displayName="Classification - Classes",
                            name="n_classes",
                            datatype="GPLong",
                            parameterType="Optional",
                            direction="Input"),
 
            arcpy.Parameter(displayName="Classification - Class Range (e.g.: 0;0.5;1;2)",
                            name="manual_breaks",
                            datatype="GPString",
                            parameterType="Optional",
                            direction="Input"),
 
            arcpy.Parameter(displayName="Classification - Underserved Threshold (A_SCL <= threshold)",
                            name="threshold",
                            datatype="GPDouble",
                            parameterType="Required",
                            direction="Input"),

            # -- Output
            arcpy.Parameter(displayName="Output Feature Class",
                            name="output_fc",
                            datatype="DEFeatureClass",
                            parameterType="Required",
                            direction="Output"),
 
            arcpy.Parameter(displayName="Save Intermediate Data",
                            name="keep_temp",
                            datatype="GPBoolean",
                            parameterType="Optional",
                            direction="Input"),
            
        ]

        params[1].parameterDependencies = ["in_fac_fc"]         # fac_id_field
        params[2].parameterDependencies = ["in_fac_fc"]         # fac_capacity_field
        params[4].parameterDependencies = ["in_demand_fc"]      # demand_id_field
        params[5].parameterDependencies = ["in_demand_fc"]      # demand_pop_field

        # filter tipe geometri / tipe field
        params[0].filter.list = ["Point"]                               # in_fac_fc
        params[3].filter.list = ["Polygon", "Point"]                    # in_demand_fc
        params[2].filter.list = ["Short", "Long", "Float", "Double"]    # capacity_field_pos
        params[5].filter.list = ["Short", "Long", "Float", "Double"]    # demand_pop_field

        # Value list + default value dari Travel Mode
        params[7].filter.type = "ValueList"                     # travel_mode
        params[7].filter.list = []

        params[8].filter.type = "ValueList"                     # dist_unit
        params[8].filter.list = ["Kilometers", "Meters"]
        params[8].value = "Kilometers"

        params[10].filter.type = "ValueList"                    # method
        params[10].filter.list = list(METHODS)
        params[10].value = "GAUSSIAN"

        params[11].value = 1.0                                  # beta
        params[11].enabled = False
        # params[12] zone_breaks: diisi pengguna saat metode ZONES
        params[12].enabled = False
        params[13].value = 10000                                # scale

        params[14].filter.type = "ValueList"                    # class_method
        params[14].filter.list = ["QUANTILE", "EQUAL INTERVAL", "NATURAL BREAKS", "MANUAL"]
        params[14].value = "NATURAL BREAKS"
        params[15].value = 5                                    # n_classes
        params[16].enabled = False                              # manual_breaks
        params[17].value = 0.0                                  # threshold
 
        params[19].value = True                                 # keep_temp

        return params

    def isLicensed(self):
        """Set whether the tool is licensed to execute."""
        try:
            return arcpy.CheckExtension("Network") == "Available"
        except Exception:
            return False


    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""

        p = _by_name(parameters)
        nd, tmode = p["network_dataset"], p["travel_mode"]
        if nd.valueAsText:
            key = _network_key(nd)
            if not getattr(nd, "hasBeenValidated", True):
                _MODE_CACHE.pop(key, None)
            try:
                modes = _travel_modes(key)
                if list(tmode.filter.list or []) != modes:
                    tmode.filter.list = modes
                if modes and tmode.value not in modes:
                    tmode.value = modes[0]
            except Exception:
                pass
        elif nd.altered:
            tmode.filter.list = []
        m = (p["method"].valueAsText or "").upper()
        p["beta"].enabled = m in ("POWER", "EXPONENTIAL")
        p["zone_breaks"].enabled = (m == "ZONES")
        if m == "ZONES":
            # d0 otomatis = batas zona terakhir
            try:
                p["d0"].value = parse_zone_breaks(p["zone_breaks"].valueAsText)[-1]
            except Exception:
                pass
            p["d0"].enabled = False
        else:
            p["d0"].enabled = True
        cm = (p["class_method"].valueAsText or "").upper()
        p["manual_breaks"].enabled = (cm == "MANUAL")
        p["n_classes"].enabled = (cm != "MANUAL")


    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""

        p = _by_name(parameters)
        for k in ("d0", "scale"):
            v = p[k].value
            if v is not None and v <= 0:
                p[k].setErrorMessage("Nilai harus > 0.")
        if p["zone_breaks"].enabled:
            try:
                parse_zone_breaks(p["zone_breaks"].valueAsText)
            except Exception as e:
                p["zone_breaks"].setErrorMessage(str(e))
        if p["manual_breaks"].enabled:
            txt = p["manual_breaks"].valueAsText
            if not txt:
                p["manual_breaks"].setErrorMessage("Isi batas kelas untuk metode MANUAL.")
            else:
                try:
                    [float(x) for x in txt.replace(",", ";").split(";") if x.strip()]
                except ValueError:
                    p["manual_breaks"].setErrorMessage("Format: angka dipisah titik koma (;).")
        if p["n_classes"].enabled and p["n_classes"].value is not None \
                and p["n_classes"].value < 2:
            p["n_classes"].setErrorMessage("Minimal 2 kelas.")
        # kontrol CRS: network harus proyeksi; facility dan demand sebaiknya ber-CRS sama
        if all(p[k].valueAsText for k in ("network_dataset", "in_fac_fc", "in_demand_fc")):
            try:
                net_sr = _network_sr(p["network_dataset"].valueAsText)
                fac_sr = arcpy.Describe(p["in_fac_fc"].valueAsText).spatialReference
                dm_sr = arcpy.Describe(p["in_demand_fc"].valueAsText).spatialReference
                owner = {"network dataset": "network_dataset", "facility": "in_fac_fc",
                         "demand": "in_demand_fc"}
                errs = _crs_errors(net_sr, fac_sr, dm_sr)
                for obj, text in errs:
                    p[owner[obj]].setErrorMessage(text)
                if not errs and not _same_sr(fac_sr, dm_sr):
                    text = ("CRS fasilitas (%s) berbeda dengan demand (%s). Saat tool dijalankan, "
                            "keduanya diproyeksikan ke CRS network (%s) sebagai data sementara."
                            % (fac_sr.name, dm_sr.name, net_sr.name))
                    p["in_fac_fc"].setWarningMessage(text)
                    p["in_demand_fc"].setWarningMessage(text)
            except Exception:
                pass          # CRS tidak dapat dibaca saat ini; diperiksa ulang saat tool dijalankan
        if p["output_fc"].valueAsText and not _output_target(p["output_fc"].valueAsText):
            p["output_fc"].setErrorMessage(
                "Output harus berupa shapefile (.shp) di dalam folder yang sudah ada, "
                "atau feature class di dalam file geodatabase (.gdb).")
            

    def execute(self, parameters, messages):
        """The source code of the tool."""

        p = _by_name(parameters)
        target = _output_target(p["output_fc"].valueAsText)
        if not target:
            raise ValueError("Output harus berupa shapefile (.shp) di folder yang sudah ada, "
                             "atau feature class di dalam geodatabase (.gdb).")
        kind, ws, out_poly = target
        raw = os.path.splitext(os.path.basename(out_poly))[0]
 
        # Lokasi data antara: geodatabase yang sama (output gdb), atau gdb kecil di folder output (shapefile)
        created_gdb = False
        if kind == "gdb":
            work = ws
        else:
            gname = arcpy.ValidateTableName(raw, ws) + "_antara.gdb"
            work = os.path.join(ws, gname)
            if not arcpy.Exists(work):
                arcpy.management.CreateFileGDB(ws, gname)
                created_gdb = True
        base = arcpy.ValidateTableName(raw, work)
        fc = os.path.join(work, base + "_fc")          # fasilitas terstandar
        dm = os.path.join(work, base + "_dm")          # titik demand terstandar
        od = os.path.join(work, base + "_od")          # tabel OD
        dm_res = os.path.join(work, base + "_dm_res")  # hasil per titik demand
        fc_res = os.path.join(work, base + "_fc_res")  # hasil per fasilitas
        hasil = os.path.join(work, base + "_hasil")    # hasil lengkap (hanya untuk output shapefile)
 
        method = p["method"].valueAsText
        zone_text = p["zone_breaks"].valueAsText
        if (method or "").upper() == "ZONES":
            d0 = parse_zone_breaks(zone_text)[-1]     # d0 = batas zona terakhir
        else:
            d0 = float(p["d0"].value)                 # cut-off catchment Step 1 dan Step 2
 
        _msg("=== [1/4] Persiapan data ===")
        run_prepare(
            p["in_fac_fc"].valueAsText, p["fac_id_field"].valueAsText,
            p["fac_capacity_field"].valueAsText, p["in_demand_fc"].valueAsText,
            p["demand_id_field"].valueAsText, p["demand_pop_field"].valueAsText,
            p["network_dataset"].valueAsText, fc, dm)
 
        _msg("=== [2/4] OD Cost Matrix (cutoff = d0 = %s %s) ===" % (
            d0, p["dist_unit"].valueAsText))
        run_od(fc, dm, p["network_dataset"].valueAsText, p["travel_mode"].valueAsText,
               d0, p["dist_unit"].valueAsText, od)
 
        _msg("=== [3/4] Perhitungan E2SFCA ===")
        run_calculate(fc, dm, od, d0, d0, method,
                      _num(p["beta"], 1.0), zone_text,
                      _num(p["scale"], 10000.0), dm_res, fc_res)
 
        _msg("=== [4/4] Klasifikasi & Laporan ===")
        report_out = out_poly if kind == "gdb" else hasil
        run_report(p["in_demand_fc"].valueAsText, p["demand_id_field"].valueAsText, dm_res,
                   p["class_method"].valueAsText, int(_num(p["n_classes"], 5)),
                   p["manual_breaks"].valueAsText, _num(p["threshold"], 0.0), report_out)
        if kind == "shp":
            _export_shapefile(hasil, out_poly)
 
        temps = [fc, dm, od, dm_res, fc_res] + ([hasil] if kind == "shp" else [])
        keep = p["keep_temp"].value
        if keep is None or keep:
            _msg("Data antara disimpan di %s: %s" % (
                work, ", ".join(os.path.basename(t) for t in temps)))
        else:
            if created_gdb:
                arcpy.management.Delete(work)          # gdb dibuat oleh tool ini
            else:
                for t in temps:
                    _delete_if_exists(t)
            _msg("Data antara dihapus.")
        _msg("Selesai. Hasil: %s" % out_poly)
        

    def postExecute(self, parameters):
        """This method takes place after outputs are processed and
        added to the display."""
        return

