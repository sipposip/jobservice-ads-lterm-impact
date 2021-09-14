using DataFrames: DataAPI
using Distributions: StatsFuns
using DataFrames
using Random
using Distributions
using StatsFuns
using GLM


function draw_data_from_influx(n, alpha_prot, maxval)
    """
    generate n individuals from the (fixed) background influx distribution
    :return: dataframe
    """
    # we have two skill features x1 and x2, and a protected feature x_prot
    # the protected feature is correlated with x2
    # we draw them from a truncated normal distribution if maxval is not None
    x_prot = rand([0,1], n)
    x1 = rand(TruncatedNormal(0,1,-maxval,maxval), n)
    x2 = 1 / 2 .* (alpha_prot .* (x_prot .- 0.5) + rand(TruncatedNormal(0,1,-maxval,maxval), n))

    s_real = compute_skill(x1, x2)
    df = DataFrame(x1= x1, x2= x2, x_prot= x_prot, s_real= s_real)
    # set T_u to 0 for all individuals
    df[!,:T_u] .= 0
    return df
end

compute_skill(x1, x2) = (x1 .+ x2) ./ 2

compute_class(df, class_boundary) = df[!,:T_u] .> class_boundary


function assign_jobs(df, loc, scale)
    """compute probability of finding a job for each individuals
    and then seperate the input individuals into those that get assigned a job
    and those that remain workless

    return: df_found_job, df_remains_workless
    """
    # in the influx population, skill is normally distributed (var=1)
    # we make the probability of finding a job a function of s_real
    # p = p('s_real'). Note that p('s_real') is NOT a probability density function of s_real, but it is a function
    # that returns the probability of the event "finds a job" given a certain value of s_real
    # we model it as a logistic function (which is automatically between [0,1])
    p  = StatsFuns.logistic.(df[!,:s_real] .* scale .- loc)
    # now select who finds a job. we can do this via drawing from a uniform distribution in [0,1]
    # and then select those where p is larger that that
    idcs_found_job = p .> rand(Uniform(0,1), length(p))
    # we copy the data to avoid potential problems with referencing etc
    df_found_job = df[idcs_found_job,:]
    df_remains_workless = df[.!idcs_found_job,:]
    @assert size(df_found_job)[1] + size(df_remains_workless)[1] == size(df)[1]
    return df_found_job, df_remains_workless
end


function train_model(df, modeltype, class_boundary)
    """
        train a logistic regression model
        for now, the classes are defined as below and above mean T_u in the training data
    """
    df[!,:class] = compute_class(df, class_boundary)
    if modeltype == "full"
        fm = @formula(class ~ x1 + x_prot)
    elseif  modeltype == "base"
        fm = @formula(class ~ x_1)
    end
    logit = glm(fm, df, Binomial(), ProbitLink())
    # drop class column again
    select!(df,Not(:class))
    return logit
end

accuracy_score(truth,pred) = mean(truth .== pred)

function recall_score(truth,pred)
    tp = sum( (truth .== 1) .& (pred .==1))
    fn = sum( (truth .== 1) .& (pred .==0))
    return tp/(tp+fn)  
end


function precision_score(truth,pred)
    tp = sum( (truth .== 1) .& (pred .==1))
    fp = sum( (truth .== 0) .& (pred .==1))
    return tp/(tp+fp)  
end


predict_class(model,df) = [if x < 0.5 0 else 1 end for x in predict(model,df)]


function main()
    # parameters
    rand_seed = 998654  # fixed random seed for reproducibility
    n_population = 10000
    alpha_prot = 2  # influence of alpha_prot on x2
    maxval = 2
    tsteps = 400  # steps after spinup
    n_spinup = 400
    n_retain_from_spinup = 200
    delta_T_u = 10
    T_u_max = 100
    modeltype = "full"
    class_boundary = 40  # in time-units
    jobmarket_function_loc = 0
    jobmarket_function_scale = 10
    # generate initial data
    # for person-pools we use dataframes, and we always use "df_" as prefix to make clear
    # that something is a pool
    df_active = draw_data_from_influx(n_population, alpha_prot, maxval)

    df_hist = DataFrame()
    df_waiting = DataFrame()
    n_waiting = 0
    model_evolution = []



    for step in 1:(n_spinup + tsteps -1)
        if step == n_spinup
            println("debug")
        end

        # assign jobs
        df_found_job, df_remains_workless = assign_jobs(df_active, jobmarket_function_loc, jobmarket_function_scale)
        n_found_job, n_remains_workless = size(df_found_job)[1], size(df_remains_workless)[1]
        df_found_job[!,:step] .= step
        # update historical data
        append!(df_hist,df_found_job)
        # increase T_u of the ones that remained workless
        df_remains_workless[!,:T_u] .+=1
        println(step)
        # at end of spinup, crop the history
        if step == n_spinup
            df_hist_start = copy(df_hist)
            model_evolution_start = copy(model_evolution)
            df_hist = df_hist[(df_hist[!,:step].>n_spinup -  n_retain_from_spinup),:]
            #model_evolution = model_evolution[-n_retain_from_spinup:end]
        end

        # remove individuals with T_u > T_u_max
        idx_remove = df_remains_workless[!,:T_u].>T_u_max
        n_removed = sum(idx_remove) # idx_remove is a boolean index, so sum gives the number of Trues
        df_remains_workless = df_remains_workless[.!idx_remove,:]

        if step > n_spinup
            # train model on all accumulated historical data
            # TODO: if we come here, df_remains_workless has length 0.....
            model = train_model(df_hist, modeltype, class_boundary)
            # group the current jobless people into the two groups
            classes = predict_class(model,df_remains_workless)



        #
            classes_true = compute_class(df_remains_workless, class_boundary)
            accur = accuracy_score(classes_true, classes)
            recall = recall_score(classes_true, classes)
            precision = precision_score(classes_true, classes)
            # here we deviate from the terminology used for the simple model.
            # since the ml-model is based on predicting the unemployment time, class 1 indicates
            # the low-prospect group (long expected unemployment time)
            df_highpros = copy(df_remains_workless[classes .== 0,:])
            df_lowpros = copy(df_remains_workless[classes .== 1,:])
            n_highpros, n_lowpros = size(df_highpros)[1], size(df_lowpros)[1]
            @assert (size(df_highpros)[1] + size(df_lowpros)[1] == size(df_remains_workless)[1])

            # TODO
            # implement intervention model here
            # END TOOO

            # for the lowpros group, we need a new attribute that describes how long they are already
            # in the waiting position, which starts at 0
            df_lowpros[!,:T_w] .= 0

            # only the highpros are retained, they will be complemented by the ones from
            # the waiting pool and by new ones later on
            df_remains_workless = df_highpros
            # move the ones that reached the final time in the waiting group to the normal
            # job seeker group

            if n_waiting > 0
                df_back_idcs = df_waiting[!,:T_w] .== delta_T_u
                df_back = df_waiting[df_back_idcs,:]
                df_waiting = df_waiting[.!df_back_idcs,:]
                if size(df_back)[1] > 0
                    df_back[!,:T_w].+=delta_T_u
                    # drop T_w
                    select!(df_back,Not(:T_w))
                    df_remains_workless = vcat(df_remains_workless, df_back)
                end
                df_waiting[!,:T_w] .+= 1
            end

            # add the new lowprps to the waiting group
            append!(df_waiting,df_lowpros)
            n_waiting = size(df_waiting)[1]
        else
            # set values to be used in the record during the spinup pahse
            accur = NaN
            precision = NaN
            recall = NaN 
        end
        # draw new people from influx to replace the ones that found a job


        # draw new people from influx to replace the ones that found a job
        df_new = draw_data_from_influx(n_found_job+n_removed, alpha_prot, maxval)
        df_active = vcat(df_remains_workless, df_new)
        n_active = size(df_active)[1]
        println(n_remains_workless)


    end
end
main()