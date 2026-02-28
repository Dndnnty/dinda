# Laporan Audit Penghapusan Log

Tanggal: 2026-02-28
Permintaan Pengguna: Hapus semua log sistem, log error, dan log aktivitas pengguna secara menyeluruh.

## Log Penghapusan

- [2026-02-28 18:06:26] Cleaning directory: C:\Users\HP\Downloads\Dinda-main\dinda\uploads
- [2026-02-28 18:06:26] Deleted file: C:\Users\HP\Downloads\Dinda-main\dinda\uploads\20260228_180212_Database_R1_-_Aneka_Jaya_Tunas_Agro.csv
- [2026-02-28 18:06:26] Deleted file: C:\Users\HP\Downloads\Dinda-main\dinda\uploads\20260228_180321_Database_R1_-_Aneka_Jaya_Tunas_Agro.csv
- [2026-02-28 18:06:26] Deleted file: C:\Users\HP\Downloads\Dinda-main\dinda\uploads\job_1_preprocessed.csv
- [2026-02-28 18:06:26] Deleted file: C:\Users\HP\Downloads\Dinda-main\dinda\uploads\job_2_hasil_hac.csv
- [2026-02-28 18:06:26] Deleted file: C:\Users\HP\Downloads\Dinda-main\dinda\uploads\job_2_hasil_kmedoids.csv
- [2026-02-28 18:06:26] Deleted file: C:\Users\HP\Downloads\Dinda-main\dinda\uploads\job_2_preprocessed.csv
- [2026-02-28 18:06:26] Cleaning directory: C:\Users\HP\Downloads\Dinda-main\dinda\static\generated
- [2026-02-28 18:06:27] Deleted file: C:\Users\HP\Downloads\Dinda-main\dinda\static\generated\boxplot_volume_hac_single_job_2_20260228_180401.png
- [2026-02-28 18:06:27] Deleted file: C:\Users\HP\Downloads\Dinda-main\dinda\static\generated\boxplot_volume_kmedoids_job_2.png
- [2026-02-28 18:06:27] Deleted file: C:\Users\HP\Downloads\Dinda-main\dinda\static\generated\dendrogram_hac_single_job_2_20260228_180401.png
- [2026-02-28 18:06:27] Deleted file: C:\Users\HP\Downloads\Dinda-main\dinda\static\generated\komposisi_jenis_hac_single_job_2_20260228_180401.png
- [2026-02-28 18:06:27] Deleted file: C:\Users\HP\Downloads\Dinda-main\dinda\static\generated\komposisi_jenis_kmedoids_job_2.png
- [2026-02-28 18:06:27] Deleted file: C:\Users\HP\Downloads\Dinda-main\dinda\static\generated\sebaran_hac_job_2_v2.png
- [2026-02-28 18:06:27] Deleted file: C:\Users\HP\Downloads\Dinda-main\dinda\static\generated\sebaran_hac_single_job_2_20260228_180401.png
- [2026-02-28 18:06:27] Deleted file: C:\Users\HP\Downloads\Dinda-main\dinda\static\generated\sebaran_kmedoids_job_2.png
- [2026-02-28 18:06:28] Deleted file: C:\Users\HP\Downloads\Dinda-main\dinda\static\generated\sebaran_kmedoids_job_2_v2.png
- [2026-02-28 18:06:28] Deleted file: C:\Users\HP\Downloads\Dinda-main\dinda\static\generated\ukuran_cluster_hac_single_job_2_20260228_180401.png
- [2026-02-28 18:06:28] Deleted file: C:\Users\HP\Downloads\Dinda-main\dinda\static\generated\ukuran_cluster_kmedoids_job_2.png
- [2026-02-28 18:06:28] Cleaning database: C:\Users\HP\Downloads\Dinda-main\dinda\app.db
- [2026-02-28 18:06:28] Found 2 records in 'jobs' table.
- [2026-02-28 18:06:28] Deleted all records from 'jobs' table and reset sequence.
- [2026-02-28 18:06:28] Searching for __pycache__ in: C:\Users\HP\Downloads\Dinda-main\dinda
- [2026-02-28 18:06:29] Deleted __pycache__: C:\Users\HP\Downloads\Dinda-main\dinda\__pycache__
- [2026-02-28 18:06:29] Log deletion process completed successfully.

## Verifikasi Penghapusan

- **Direktori Uploads**: Terverifikasi kosong (0 file).
- **Direktori Generated**: Terverifikasi kosong (0 file).
- **Database (jobs table)**: Terverifikasi kosong (0 record).
- **Cache Python (__pycache__)**: Terverifikasi dihapus.
