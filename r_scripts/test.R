source('full_admg_learning.R')

#Generate custom pag1
pag1 <- matrix(rep(0,9), nrow = 3)
pag1[2,1] <- pag1[1,2] <- 1
pag1[2,3] <- pag1[3,2] <- 1
pag1 <- make_pag_from_amat(pag1)

#Convert pag1 to the full set of ADMGs using pag2admg
admg_list1 <- pag2admg(pag1)


#Save plots and matrices to files
cat("Saving plots and matrices to files...\n")

# Save PAG plots
save_pag_plot(pag1, "test/pag1.png")


# Save ADMG matrices as text files
save_admg_matrices(admg_list1, prefix = "test/pag1_admg")

cat("All plots and matrices saved successfully!\n")