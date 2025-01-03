# ncu --set full -k "regex:.*fwd.*" -f -o output/pbs-1-pcs-16384-pcl-16384-dbs-250-dcl-12288-fused.ncu-rep python tests/banner_fig_profile.py --p_bs 1 --p_cs 16384 --p_cl 16384 --d_bs 250 --d_cl 12288 --stage fused
# ncu --set full -k "regex:.*fwd.*" -f -o output/pbs-1-pcs-12288-pcl-12288-dbs-220-dcl-12288-fused.ncu-rep python tests/banner_fig_profile.py --p_bs 1 --p_cs 12288 --p_cl 12288 --d_bs 220 --d_cl 12288 --stage fused
# ncu --set full -k "regex:.*fwd.*" -f -o output/pbs-1-pcs-1024-pcl-12288-dbs-80-dcl-12288-fused.ncu-rep python tests/banner_fig_profile.py --p_bs 1 --p_cs 1024 --p_cl 12288 --d_bs 80 --d_cl 12288 --stage fused

# ncu --set full -k "regex:.*fwd.*" -f -o output/pbs-1-pcs-16384-pcl-16384-dbs-250-dcl-12288-.ncu-rep python tests/banner_fig_profile.py --p_bs 1 --p_cs 16384 --p_cl 16384 --d_bs 250 --d_cl 12288 --stage fused
# ncu --set full -k "regex:.*fwd.*" -f -o output/pbs-1-pcs-12288-pcl-12288-dbs-220-dcl-12288-fused.ncu-rep python tests/banner_fig_profile.py --p_bs 1 --p_cs 12288 --p_cl 12288 --d_bs 220 --d_cl 12288 --stage fused
# ncu --set full -k "regex:.*fwd.*" -f -o output/pbs-1-pcs-1024-pcl-12288-dbs-80-dcl-12288-fused.ncu-rep python tests/banner_fig_profile.py --p_bs 1 --p_cs 1024 --p_cl 12288 --d_bs 80 --d_cl 12288 --stage fused


TEST_CMD="python tests/banner_fig_profile.py"
NCU_METRICS="gpu__time_duration.sum"

d_cl=4096
p_cls="512 1024 2048 4096 8192 12200 16384 20480"
d_batch_sizes="4 8 16 32 64 128 256 512"
for p_cl in $p_cls; do
    for d_bs in $d_batch_sizes; do
        echo "Running prefill with p_cl=$p_cl and d_bs=$d_bs"
        ncu --metrics $NCU_METRICS -k "regex:.*fwd.*" --csv $TEST_CMD --stage fused --p_bs 1 --p_cl $p_cl --p_cs $p_cl --d_bs $d_bs --d_cl $d_cl > output/fused-p-$p_cl-d-$d_bs-8192.csv
        ncu --metrics $NCU_METRICS -k "regex:.*fwd.*" --csv $TEST_CMD --stage prefill --p_bs 1 --p_cl $p_cl --p_cs $p_cl --d_bs $d_bs --d_cl $d_cl > output/prefill-p-$p_cl-d-$d_bs-8192.csv
        ncu --metrics $NCU_METRICS -k "regex:.*fwd.*" --csv $TEST_CMD --stage decode --p_bs 1 --p_cl $p_cl --p_cs $p_cl --d_bs $d_bs --d_cl $d_cl > output/decode-p-$p_cl-d-$d_bs-8192.csv
    done
done