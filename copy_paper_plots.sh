mkdir -p Figures
for m in 620 440 380 560 500 680 870 260; do mkdir -p Figures/benchmark_M$m; done
for m in 620 440 380 560 500 680 870 260; do mkdir -p Figures/benchmark_M${m}_recocuts; done
mkdir -p Figures/nlo_vs_lo
mkdir -p Figures/validation

for m in 380 440 500 560 620 680 870; do
  cp plots_CompWOInt/dihiggs_LO_CompAll_hh_mass_BM_singlet_M${m}.pdf Figures/benchmark_M${m}/.
  cp plots_CompWOInt_recocuts/dihiggs_LO_CompAll_hh_mass_smear_improved_paired_BM_singlet_M${m}.pdf Figures/benchmark_M${m}_recocuts/.

done

cp plots_CompWOInt/dihiggs_LO_CompAll_hh_mass_BM_singlet_M260_logy.pdf Figures/benchmark_M260/. 
cp plots_CompWOInt/dihiggs_LO_CompNonRes_hh_mass_BM_singlet_M260.pdf Figures/benchmark_M260/. 
cp plots_CompWOInt_recocuts/dihiggs_LO_CompAll_hh_mass_smear_improved_paired_BM_singlet_M260_logy.pdf Figures/benchmark_M260_recocuts/.

cp plots_CompWOInt_recocuts/dihiggs_LO_CompAll_hh_mass_smear_improved_optimistic_BM_singlet_M380.pdf Figures/benchmark_M380_recocuts/.

for var in hh_pT h1_pT h2_pT b1_pT b4_pT HT hh_eta hh_deta hh_dphi hh_dR; do
  cp plots_CompWOInt/dihiggs_LO_CompAll_${var}_BM_singlet_M620.pdf Figures/benchmark_M620/. 
  cp plots_CompWOInt_recocuts/dihiggs_LO_CompAll_${var}_smear_BM_singlet_M620.pdf Figures/benchmark_M620_recocuts/.

done

for m in 380 440 500 560 620 680 870 260; do
  cp plots_NLO/dihiggs_ReweightValidation_noBefore_hh_mass_fineBins_singlet_M${m}.pdf Figures/validation/.
done

cp plots_NLO/dihiggs_ReweightValidation_hh_mass_fineBins_singlet_M600.pdf Figures/validation/.
cp plots_NLO/dihiggs_ReweightValidation_h1_pT_singlet_M600.pdf Figures/validation/.


cp plots_NLO/dihiggs_LO_contributions_hh_mass_fineBins.pdf Figures/validation/.
cp plots_NLO/dihiggs_LO_WidthComp_box_SH_hh_mass_fineBins.pdf Figures/validation/.


cp plots_NLO/dihiggs_NLO_vsLO{_,_inc_kfactors_}hh_mass_{box,Sh,SP,box_Sh_int,box_SP,SP_Sh}.pdf Figures/nlo_vs_lo/.

cp plots_NLO/dihiggs_NLO_vsLO_inc_kfactors_hh_pT_{box,Sh,SP,box_Sh_int}.pdf Figures/nlo_vs_lo/.
