# E2SFCA – Toolbox Aksesibilitas Spasial untuk ArcGIS Pro

Python toolbox (`.pyt`) untuk menghitung **aksesibilitas spasial** fasilitas layanan dengan metode **Enhanced Two-Step Floating Catchment Area (E2SFCA)** (Luo & Qi, 2009). Seluruh tahapan berjalan dalam **satu tool dan satu kali run**: Persiapan data, OD Cost Matrix berbasis jarak jaringan jalan, perhitungan E2SFCA, lalu klasifikasi dan penandaan wilayah *underserved*.

## Fitur

- Jarak dihitung dengan **jaringan jalan** (Network Dataset, OD Cost Matrix, impedansi **jarak**, bukan waktu).
- Demand berupa **poligon** (dihitung centroid-nya otomatis) atau **titik**; populasi dibaca dari atribut.
- Kapasitas fasilitas bersifat opsional (default = 1).
- Lima fungsi *distance decay*: `GAUSSIAN`, `ZONES`, `BINARY` (2SFCA standar, pembanding), `POWER`, `EXPONENTIAL`.
- Satu jarak cut-off `d0` yang dipakai di Step 1 dan Step 2.
- Penyamaan CRS otomatis ke CRS network dataset (data sementara), termasuk pemilihan transformasi datum jika datum berbeda.
- Klasifikasi hasil: `NATURAL BREAKS` (Jenks, default), `QUANTILE`, `EQUAL INTERVAL`, atau `MANUAL`, plus penanda wilayah *underserved* berdasarkan ambang.
- Output ke **geodatabase** atau **shapefile**.

## Persyaratan

| Kebutuhan | Keterangan |
|---|---|
| ArcGIS Pro | dengan `arcpy`, termasuk modul `arcpy.nax`. |
| Ekstensi **Network Analyst** | wajib |
| Network Dataset | ber-CRS **proyeksi**, dengan travel mode yang impedansinya berupa **atribut jarak** |
| Python | `numpy` dan `pandas` (sudah termasuk dalam lingkungan Python bawaan ArcGIS Pro) |

## Instalasi

1. Salin `E2SFCA.pyt` dan `E2SFCA.E2SFCA.pyt.xml` ke **satu folder** (XML berisi teks bantuan parameter; tool tetap berjalan tanpa XML).
2. Di ArcGIS Pro, panel **Catalog** → klik kanan **Toolboxes** → **Add Toolbox**, lalu pilih `E2SFCA.pyt`.
3. Buka tool **E2SFCA** dari toolbox tersebut.

> Nama file XML harus `E2SFCA.E2SFCA.pyt.xml` (pola `<nama toolbox>.<nama kelas tool>.pyt.xml`) dan berada di folder yang sama dengan `.pyt`.

## Data masukan

| Data | Syarat |
|---|---|
| **Facility** | layer **titik**. Field ID unik (opsional; kosong = ObjectID). Field kapasitas numerik (opsional; kosong, null, atau ≤ 0 = 1) |
| **Demand** | layer **poligon** atau **titik**. Field ID **unik dan tidak kosong**; field populasi numerik (kosong atau negatif dianggap 0) |
| **Network Dataset** | ber-CRS proyeksi; travel mode berimpedansi jarak |

Satu wilayah yang terdiri atas beberapa record (mis. pulau terpisah) harus digabung lebih dulu per ID (mis. dengan *Dissolve*), karena ID ganda menghentikan proses.

## Parameter

| # | Parameter | Tipe | Keterangan |
|---|---|---|---|
| 0 | **Facility Location** | Feature Layer (Point) | titik lokasi fasilitas |
| 1 | Facility – ID Field | Field | opsional; default ObjectID |
| 2 | Facility – Capacity Field | Field (numerik) | opsional; default 1 |
| 3 | **Demand Location** | Feature Layer (Polygon/Point) | poligon dikonversi ke centroid |
| 4 | **Demand – ID Field** | Field | wajib; harus unik |
| 5 | **Demand – Population Field** | Field (numerik) | wajib |
| 6 | **Network Dataset** | Network Dataset Layer | wajib; CRS proyeksi |
| 7 | **Network Dataset – Travel Mode** | String | daftar terisi otomatis dari network |
| 8 | **Network Dataset – Distance Unit** | String | `Kilometers` (default) atau `Meters`; satuan `d0` dan tabel jarak |
| 9 | **E2SFCA – Distance Threshold (d0)** | Double | radius catchment; otomatis = batas zona terakhir untuk `ZONES` |
| 10 | **E2SFCA – Distance Decay Function** | String | `GAUSSIAN` (default), `ZONES`, `BINARY`, `POWER`, `EXPONENTIAL` |
| 11 | E2SFCA – Beta | Double | untuk `POWER` / `EXPONENTIAL` (default 1) |
| 12 | E2SFCA – Zone Break | String | untuk `ZONES`; batas atas tiap zona, mis. `5;10;15` |
| 13 | **E2SFCA – Result Scale Factor** | Double | default 10000 |
| 14 | **Classification Method** | String | `NATURAL BREAKS` (default), `QUANTILE`, `EQUAL INTERVAL`, `MANUAL` |
| 15 | Classification – Classes | Long | default 5; tidak dipakai untuk `MANUAL` |
| 16 | Classification – Class Range | String | untuk `MANUAL`; batas atas kelas, mis. `0;0.5;1;2` |
| 17 | **Classification – Underserved Threshold** | Double | default 0; underserved jika `A_SCL <= ambang` |
| 18 | **Output Feature Class** | Feature Class | `.shp` di folder yang sudah ada, atau feature class di `.gdb` |
| 19 | Save Intermediate Data | Boolean | default `True` |

Parameter bercetak tebal wajib diisi.

## Metode

Untuk setiap fasilitas *j* dan titik demand *i*, dengan jarak jaringan *d* dan fungsi bobot *W(d)*:

```
Step 1:  R_j = S_j / Σ_k ( P_k · W(d_kj) )      untuk d_kj ≤ d0
Step 2:  A_i = Σ_j ( R_j · W(d_ij) )            untuk d_ij ≤ d0
```

`S_j` = kapasitas fasilitas, `P_k` = populasi demand, `R_j` = rasio fasilitas terhadap populasi terbobot, `A_i` = indeks aksesibilitas. Bobot selalu 0 untuk `d > d0`.

### Fungsi distance decay

| Fungsi | Rumus / perilaku |
|---|---|
| `GAUSSIAN` | `(exp(-½(d/d0)²) − exp(-½)) / (1 − exp(-½))`; W(0) = 1, W(d0) = 0 |
| `ZONES` | jarak dibagi zona menurut batas yang diisi; bobot tiap zona = Gaussian ternormalisasi di titik tengah zona, dengan skala d0 = batas terakhir; konstan di dalam zona |
| `BINARY` | bobot 1 di dalam d0: setara 2SFCA standar, cocok sebagai pembanding |
| `POWER` | `(max(d, 0.05·d0) / (0.05·d0))^(−β)` |
| `EXPONENTIAL` | `exp(−β · d / d0)` |

Pada `ZONES`, isi **jarak**, bukan bobot. Contoh `5;10;15` berarti zona 0–5, 5–10, dan 10–15 (dalam satuan jarak yang dipilih); nilai terakhir menjadi `d0`.

## Alur proses

```
[1/4] Persiapan data   → periksa ID, samakan CRS ke network, hitung centroid demand
[2/4] OD Cost Matrix   → jarak jaringan fasilitas–demand, cutoff = d0
[3/4] Perhitungan      → E2SFCA Step 1 dan Step 2
[4/4] Klasifikasi      → kelas aksesibilitas dan penanda underserved
```

**Penanganan CRS.** Network dataset harus berCRS proyeksi dan menjadi acuan. Layer fasilitas dan demand yang CRS-nya berbeda diproyeksikan ke CRS network sebagai data sementara (layer asli tidak diubah). Jika datum berbeda, transformasi datum dipilih otomatis dari `arcpy.ListTransformations` (kandidat pertama), dan nama transformasi dicatat di pesan proses. Dialog memberi peringatan jika CRS fasilitas dan demand berbeda.

**Centroid.** Centroid poligon yang jatuh di luar poligonnya (poligon cekung, pesisir, kepulauan) dipindahkan ke *label point* di dalam poligon.

**Satuan jarak.** `d0` dalam `Distance Unit` dikonversi otomatis ke satuan impedansi network sebelum dikirim ke solver, lalu jarak hasil dikonversi kembali. Tool berhenti dengan error jika impedansi bukan atribut jarak atau satuannya tidak terbaca.

**Titik di luar jaringan.** Fasilitas dan demand yang tidak dapat dipakai solver (tidak terlokasi di jaringan) dilaporkan di pesan proses, lengkap dengan contoh ID dan total populasi yang terdampak. Jika tidak ada satu pun pasangan fasilitas–demand dalam `d0`, tool berhenti dengan penjelasan penyebabnya.

## Keluaran

**Output Feature Class** adalah salinan layer demand (atribut aslinya tetap ada) dengan field tambahan:

| Field | Isi |
|---|---|
| `POP` | populasi |
| `A_SCL` | aksesibilitas × faktor skala (dengan S = 1 dan skala 10000: fasilitas per 10.000 penduduk). Makin besar makin baik; 0 = tidak ada fasilitas dalam `d0` |
| `N_FC` | jumlah fasilitas dalam radius `d0` dari wilayah tersebut |
| `A_CLS` | kelas aksesibilitas; 1 = terendah |
| `UNDERSRV` | 1 jika `A_SCL ≤ ambang`, selain itu 0 |


**Output shapefile.** Shapefile tidak menyimpan NULL, sehingga wilayah tidak dinilai diberi kode `A_SCL = -9999`, `A_CLS = 0`, `UNDERSRV = -1`. Nama field asli yang lebih dari 10 karakter dipotong dan diberi angka jika bentrok; daftarnya dicatat di pesan proses.

**Data antara** (jika *Save Intermediate Data* dicentang) disimpan di geodatabase yang sama dengan output, dengan akhiran `_fc`, `_dm`, `_od`, `_dm_res`, `_fc_res`. Untuk output shapefile, dibuat geodatabase `<nama>_antara.gdb` di folder yang sama (ditambah `_hasil`, layer hasil lengkap sebelum diekspor).

## Catatan penggunaan

- **Field populasi** harus benar-benar berisi jumlah penduduk.
- **Pemilihan `d0`** menentukan hasil secara langsung; sesuaikan dengan konteks layanan (mis. jarak tempuh yang wajar ke jangkauan layanan) dan bandingkan beberapa nilai.
- **Natural Breaks** pada data yang sangat besar dihitung dari sampel 3.000 kuantil agar tetap cepat.
- **Akurasi jarak** bergantung pada atribut panjang (`Length`) pada network dataset.
- `canRunInBackground = False`: tool berjalan di proses utama ArcGIS Pro.

## Status dan keterbatasan

- Bagian yang bergantung pada ArcGIS (proyeksi, centroid, OD Cost Matrix, ekspor shapefile) perlu divalidasi pada data Anda sendiri; mulailah dengan data kecil.
- Transformasi datum dipilih otomatis (kandidat pertama) dan tidak dapat diubah dari dialog.
- Hanya satu jarak cut-off untuk Step 1 dan Step 2.
- Bobot zona dihitung dari jarak yang Anda isi dengan fungsi Gaussian; tool tidak menerima bobot zona yang ditentukan sendiri.

## Struktur repositori

```
.
├── E2SFCA.pyt                  # toolbox dan tool E2SFCA
├── E2SFCA.E2SFCA.pyt.xml       # teks bantuan parameter (metadata)
├── README.md
└── LICENSE
```

## Referensi

- Luo, W., & Wang, F. (2003). Measures of spatial accessibility to health care in a GIS environment: synthesis and a case study in the Chicago region. *Environment and Planning B*, 30(6), 865–884. (2SFCA)
- Luo, W., & Qi, Y. (2009). An enhanced two-step floating catchment area (E2SFCA) method for measuring spatial accessibility to primary care physicians. *Health & Place*, 15(4), 1100–1107. (E2SFCA)
- *JMIR Formative Research* 2024

## Lisensi

-

## Penulis

Tedy Imanuel (https://www.linkedin.com/in/tedy-selan/)
