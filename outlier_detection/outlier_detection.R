library(data.table)
library(ggplot2)
pheno_data <- fread("data/train/train.csv")

pheno_data_outliers <- copy(pheno_data)

zscore <- function(x){(x-mean(x))/sd(x)}

quality_scores <- c("anat_cnr","anat_efc","anat_fber", "anat_fwhm", "anat_qi1","anat_snr", 
                    "func_efc","func_fber", "func_fwhm", "func_dvars", "func_outlier",
                    "func_quality", "func_mean_fd", "func_num_fd","func_perc_fd", "func_gsr")


pheno_data.melt <- melt(pheno_data,measure.vars = quality_scores)
pdf("quality_score.pdf")
ggplot(pheno_data.melt,aes(x=value)) + geom_histogram() + facet_wrap(. ~ variable, scales = "free") + theme_classic()
dev.off()

pheno_data_outliers[, c(quality_scores) := lapply(.SD,zscore) ,.SDcols=quality_scores]

pheno_data_outliers.melt <- melt(pheno_data_outliers,measure.vars = quality_scores)
pdf("quality_zscore.pdf")
ggplot(pheno_data_outliers.melt,aes(x=value)) + geom_histogram() + facet_wrap(. ~ variable, scales = "free") + theme_classic()
dev.off()

res <- lapply(quality_scores,function(x){
        pheno_data_outliers[get(x) > 3]$fn_image_txt
        })
names(res) <- quality_scores

n_outliers <- data.table(n_outliers=unlist(lapply(res,length)))
ggplot(n_outliers,aes(x=n_outliers)) + geom_histogram(binwidth = 1) + theme_classic() 

outliers <- unlist(res)
outlier_frequency <- data.table(outlier_frequency = as.vector(table(outliers)))
ggplot(outlier_frequency,aes(x=outlier_frequency)) + geom_histogram(binwidth = 1) + theme_classic() 

unique_outliers <- unique(outliers)
length(unique_outliers)

write.table(data.table(outliers=unique_outliers),"unique_outliers.txt",sep="\t",quote=F,row.names = F,col.names = T)
