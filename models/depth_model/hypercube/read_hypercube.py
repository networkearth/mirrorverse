sql = """
with base as (
    select 
        _decision,
        _choice,
        depth_class * std_depth_class + avg_depth_class as depth_class,
        period_progress * std_period_progress + avg_period_progress as period_progress,
        elevation * std_elevation + avg_elevation as elevation,
        month * std_month + avg_month as month,
        daytime
    from 
        depth_model_hypercube_mk3
), probs as (
    select 
        _decision,
        _choice,
        probability
    from
        depth_model_mk3_hypercube_inference
)
select 
    base.*,
    probs.probability as probability
from
    base
    inner join probs 
        on base._decision = probs._decision 
        and base._choice = probs._choice

"""
hypercube = db.read_data(sql)
print(hypercube.shape)
hypercube.head()