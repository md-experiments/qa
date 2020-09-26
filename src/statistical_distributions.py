from scipy.special import gamma

def beta_parameter_estimation(x,v):
    alpha_hat=x*((x*(1-x))/v-1)
    beta_hat=(1-x)*((x*(1-x))/v-1)
    return alpha_hat,beta_hat

def beta_pdf(x,alpha,beta):
    big_b=(gamma(alpha)*gamma(beta))/gamma(alpha+beta)
    return (x**(alpha-1))*((1-x)**(beta-1))/big_b

def beta_pdf_vector(mean,var,count):
    alpha_hat,beta_hat=beta_parameter_estimation(mean,var)

    pdf_vec=[count*beta_pdf(x/100,alpha_hat,beta_hat) for x in range(0,101)]
    return pdf_vec/(sum(pdf_vec)/count)