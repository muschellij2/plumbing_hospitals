## ----setup, include = FALSE---------------------------------------------------
library(knitr)
library(analogsea)
library(distribglm)
knitr::opts_chunk$set(echo = FALSE, message = FALSE, comment = "")
library(dplyr)
library(broom)
library(ggplot2)
set.seed(20201009)


## ----fcap, echo = FALSE-------------------------------------------------------
workflow_cap = paste0("Illustration of the overall workflow.  ", 
                      "An analyst sets up the model using the setup\\_model function.  ", 
                      "Once that is done, each respective site reads in their data and then runs estimate\\_model, which computes gradients and sends them to the folder/server.  ", 
                      "Once all gradient values are computed, the coefficients for the next iterations are returned and the gradient is computed again.  ", 
                      "This process is repeated until convergence or a fixed number of iterations.  ",
                      "The final model is then located in the synced folder/server, can be downloaded, and the iterations can either be deleted or investigated to ensure convergence and private information has been secure."
)


## ----workflow, fig.cap = workflow_cap,  echo = FALSE--------------------------
knitr::include_graphics("workflow.png", dpi = NA)


## ---- eval =  FALSE, echo = TRUE----------------------------------------------
## library(distribglm)
## setup_model(model_name = "death_age_sex",
##             formula = "death ~ age + sex",
##             family = binomial(),
##             all_site_names = c("site1", "site2", "site3"),
##             synced_folder = "~/Dropbox/shared_folder")


## ---- eval =  FALSE, echo = TRUE----------------------------------------------
## library(distribglm)
## site2_data = read.csv("/path/to/mortality_data.csv")
## estimate_model(model_name = "death_age_sex",
##                site_name = "site2",
##                data = site2_data,
##                synced_folder = "C:/Dropbox/my_shared_folder")


## ---- eval =  FALSE-----------------------------------------------------------
## compute_model(model_name = "death_age_sex",
##               synced_folder = "~/Dropbox/shared_folder")


## ---- eval = FALSE------------------------------------------------------------
## droplet = analogsea::droplet_create()
## droplet = distribglm::do_deploy_glm_api(droplet, r_packages = "dplyr", github_r_packages = c("muschellij2/distribglm"))
## droplet


## ---- eval = FALSE, echo = TRUE-----------------------------------------------
## droplet


## ---- eval = TRUE, echo = FALSE-----------------------------------------------
droplet = structure(list(id = 210324177L, name = "TougherAviation", memory = 1024L, 
                         vcpus = 1L, disk = 25L, locked = FALSE, status = "active", 
                         kernel = NULL, created_at = "2020-10-02T15:56:50Z", features = list(
                           "private_networking"), backup_ids = list(), next_backup_window = NULL, 
                         snapshot_ids = list(), image = list(id = 69439389L, name = "18.04 (LTS) x64", 
                                                             distribution = "Ubuntu", slug = "ubuntu-18-04-x64", public = TRUE, 
                                                             regions = list("nyc3", "nyc1", "sfo1", "nyc2", "ams2", 
                                                                            "sgp1", "lon1", "ams3", "fra1", "tor1", "sfo2", "blr1", 
                                                                            "sfo3"), created_at = "2020-09-02T19:36:10Z", min_disk_size = 15L, 
                                                             type = "base", size_gigabytes = 2.36, description = "Ubuntu 18.04 x86 image", 
                                                             tags = list(), status = "available"), volume_ids = list(), 
                         size = list(slug = "s-1vcpu-1gb", memory = 1024L, vcpus = 1L, 
                                     disk = 25L, transfer = 1, price_monthly = 5, price_hourly = 0.00744, 
                                     regions = list("ams2", "ams3", "blr1", "fra1", "lon1", 
                                                    "nyc1", "nyc2", "nyc3", "sfo1", "sfo2", "sfo3", "sgp1", 
                                                    "tor1"), available = TRUE), size_slug = "s-1vcpu-1gb", 
                         networks = list(v4 = list(list(ip_address = "10.120.0.2", 
                                                        netmask = "255.255.240.0", gateway = "<nil>", type = "private"), 
                                                   list(ip_address = "64.225.124.41", netmask = "255.255.240.0", 
                                                        gateway = "64.225.112.1", type = "public")), v6 = list()), 
                         region = list(name = "San Francisco 2", slug = "sfo2", features = list(
                           "private_networking", "backups", "ipv6", "metadata", 
                           "install_agent", "storage", "image_transfer"), available = TRUE, 
                           sizes = list("s-1vcpu-1gb", "512mb", "s-1vcpu-2gb", "1gb", 
                                        "s-3vcpu-1gb", "s-2vcpu-2gb", "s-1vcpu-3gb", "s-2vcpu-4gb", 
                                        "2gb", "s-4vcpu-8gb", "m-1vcpu-8gb", "c-2", "4gb", 
                                        "c2-2vcpu-4gb", "g-2vcpu-8gb", "gd-2vcpu-8gb", "m-16gb", 
                                        "s-8vcpu-16gb", "s-6vcpu-16gb", "c-4", "8gb", "c2-4vpcu-8gb", 
                                        "m-2vcpu-16gb", "m3-2vcpu-16gb", "g-4vcpu-16gb", 
                                        "gd-4vcpu-16gb", "m6-2vcpu-16gb", "m-32gb", "s-8vcpu-32gb", 
                                        "c-8", "16gb", "c2-8vpcu-16gb", "m-4vcpu-32gb", "m3-4vcpu-32gb", 
                                        "g-8vcpu-32gb", "s-12vcpu-48gb", "gd-8vcpu-32gb", 
                                        "m6-4vcpu-32gb", "m-64gb", "s-16vcpu-64gb", "c-16", 
                                        "32gb", "c2-16vcpu-32gb", "m-8vcpu-64gb", "m3-8vcpu-64gb", 
                                        "g-16vcpu-64gb", "s-20vcpu-96gb", "48gb", "gd-16vcpu-64gb", 
                                        "m6-8vcpu-64gb", "m-128gb", "s-24vcpu-128gb", "c-32", 
                                        "64gb", "c2-32vpcu-64gb", "m-16vcpu-128gb", "m3-16vcpu-128gb", 
                                        "g-32vcpu-128gb", "s-32vcpu-192gb", "gd-32vcpu-128gb", 
                                        "m-224gb", "m6-16vcpu-128gb", "g-40vcpu-160gb", "gd-40vcpu-160gb")), 
                         tags = list(), vpc_uuid = "5243aa8a-2d90-46d9-9be8-686db7a6a9bb"), class = "droplet")
print(droplet)


## ---- echo = FALSE, results="hide"--------------------------------------------
ips = do.call("rbind", lapply(droplet$networks$v4, as.data.frame))
ips

## ---- echo = FALSE------------------------------------------------------------
ip = paste0("https://", ips$ip_address[ips$type == "public"])


## ---- eval = FALSE, echo = TRUE-----------------------------------------------
## api_set_url(url = paste0(ip, "/glm"))


## ---- eval = FALSE, echo = TRUE-----------------------------------------------
## setup_model = api_setup_model(
##   model_name = "death_age_sex_api_model",
##   formula = "death ~ age + sex", family = "binomial",
##   link = "logit",
##   all_site_names = c("site1", "site2", "site3"),
##   tolerance = 1e-12)


## ---- eval = FALSE, echo = TRUE-----------------------------------------------
## auth_hdr = httr::add_headers(
##   Authorization = paste0("Key ", Sys.getenv("CONNECT_API_KEY")))
## setup_model = api_setup_model(..., config = auth_hdr)


## ---- eval = FALSE, echo = TRUE-----------------------------------------------
## site2_data = read.csv("/path/to/mortality_data.csv")
## api_estimate_model(model_name =  "death_age_sex_api_model",
##                    site_name = "site2", data = site2_data)


## ----dcap, echo = FALSE-------------------------------------------------------
docs_cap = paste0("Snapshot of the API documentation.  This API endpoint documentation", 
                  " is automatically created when deployed, including the parameters needed for", 
                  " each endpoint.")


## ----docs, fig.cap = docs_cap, echo = FALSE-----------------------------------
knitr::include_graphics("docs.png", dpi = NA)


## ----gendata, echo = TRUE-----------------------------------------------------
generate_data = function(
  n = 1000, n_sites = 4,
  all_site_names = paste0("site", 1:n_sites)) {
  
  probs = runif(n = n_sites)
  probs = probs / sum(probs)
  indices = sample(
    1:n_sites, size = n,
    prob = probs, replace = TRUE)
  
  true_beta = c(0.25, 1.25, -0.3)
  df = data.frame(
    ones = rep(1, n),
    x1 = rnorm(n),
    x2 = rnorm(n, mean = 0, sd = 2)
  )
  df$x1 = df$x1 + rnorm(n, mean = indices)
  expb = exp(as.matrix(df) %*% true_beta )
  df$ones = NULL
  df$prob_y = expb/(1 + expb)
  df$y = rbinom(n, size = 1, prob = df$prob_y)
  df$lambda = drop(expb)
  df$pois_y = rpois(n, lambda = df$lambda)
  df$site = indices

  df
}
df = generate_data()
head(df)


## ---- echo = TRUE-------------------------------------------------------------
true_model = glm(y ~ x1 + x2, data = df, family = binomial())


## ---- echo = TRUE-------------------------------------------------------------
datasets = split(df, df$site)


## ---- echo = TRUE-------------------------------------------------------------
model_name = "simple_logistic"
api_setup_model(model_name = model_name, formula = y ~ x1 + x2,
                all_site_names = paste0("site", 1:4), tolerance = 1e-12)


## ----runreg, echo = TRUE------------------------------------------------------
for (i in 1:10) {
  for (site_number in 1:4) {
    data = datasets[[as.character(site_number)]]
    site_name = paste0("site", site_number)
    api_submit_gradient(model_name = model_name, verbose = FALSE, 
                        data = data,
                        site_name = site_name)
  }
  beta = api_get_current_beta(model_name = model_name)
  if (beta$converged) {
    break
  }
}


## ----model_output, echo = TRUE------------------------------------------------
model = api_model_converged(model_name)
model


## ---- echo = TRUE-------------------------------------------------------------
true_model


## ----covar, echo = TRUE-------------------------------------------------------
max(abs(vcov(true_model) - model$covariance))


## ---- echo = TRUE-------------------------------------------------------------
true_summary = summary(true_model)
true_summary$coefficients
model$z_value


## ----trace, echo = TRUE-------------------------------------------------------
model_trace = api_model_trace(model_name)


## ----tpfigcap, echo = FALSE---------------------------------------------------
traceplot_cap = paste0("Plot of coefficients over iterations. ",
                      "We present the coefficients over a number of iterations, separated by each variable.  The blue line represents the estimate of the coefficient from the full-dataset model, which we are aiming to estimate.")


## ----traceplot, fig.cap = traceplot_cap, echo = FALSE-------------------------
coefs = as.data.frame(sapply(model_trace, `[[`, "coefficients"))
coefs$term = model_trace[[1]]$beta_names
coefs = tidyr::gather(coefs, var, value, -term)
coefs = coefs[grepl("iteration", coefs$var), ]
coefs$index = as.numeric(factor(coefs$var, levels = unique(coefs$var)))
true_coef = broom::tidy(true_model) %>% select(term, estimate)
coefs %>% 
  ggplot(aes(x = index, y = value)) + geom_point() + geom_line() + 
  facet_wrap(~ term, ncol = 1, scales = "free_y") + 
  geom_hline(data = true_coef, aes(yintercept = estimate), col = "blue") + 
  xlab("Iteration") + ylab("Coefficient")

