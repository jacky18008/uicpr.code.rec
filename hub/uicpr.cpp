#define _GLIBCXX_USE_CXX11_ABI 1
#include <omp.h>
#include <sstream>
#include "../src/util/util.h"                       // arguments
#include "../src/util/file_graph.h"                 // graph
#include "../src/sampler/vc_sampler.h"              // sampler
#include "../src/mapper/lookup_mapper.h"            // mapper
#include "../src/optimizer/triplet_optimizer.h"     // optimizer

int main(int argc, char **argv){

    // arguments
    ArgParser arg_parser(argc, argv);
    std::string train_ui_path = arg_parser.get_str("-train_ui", "user-item.graph", "input user-item graph path");
    std::string train_ui_neg_path = arg_parser.get_str("-train_ui_neg", "user-item.graph", "input user-item graph path");
    std::string train_up_path = arg_parser.get_str("-train_up", "user-profile.graph", "input user-profile graph path");
    std::string train_im_path = arg_parser.get_str("-train_im", "item-meta.graph", "input item-meta graph path");
    std::string save_name_q = arg_parser.get_str("-save_q", "uicpr.ui.embed", "path for saving mapper as query");
    std::string save_name_k = arg_parser.get_str("-save_k", "uicpr.meta.embed", "path for saving mapper as key");
    int dimension = arg_parser.get_int("-dimension", 64, "embedding dimension");
    int num_negative = arg_parser.get_int("-num_negative", 5, "number of negative sample");
    double update_times = arg_parser.get_double("-update_times", 10, "update times (*million)");
    double init_alpha = arg_parser.get_double("-init_alpha", 0.1, "init learning rate");
    double user_reg = arg_parser.get_double("-user_reg", 0.01, "l2 regularization");
    double item_reg = arg_parser.get_double("-item_reg", 0.01, "l2 regularization");
    int worker = arg_parser.get_int("-worker", 1, "number of worker (thread)");

    if (argc == 1) {
        return 0;
    }

    // 0. [FileGraph] read graph
    std::cout << "(UI-Graph)" << std::endl;
    FileGraph ui_file_graph(train_ui_path, 0);
    FileGraph ui_file_graph_neg(train_ui_neg_path, 0, ui_file_graph.index2node);
    std::vector<FileGraph> up_file_graph;
    std::vector<FileGraph> im_file_graph;
    std::string sub_path;

    std::cout << "(UP-Graph)" << std::endl;
    std::stringstream ss;
    ss << train_up_path;
    while(ss.good())
    {
        getline(ss, sub_path, ',');
        if (up_file_graph.size())
            up_file_graph.push_back(FileGraph(sub_path, 0, up_file_graph.back().index2node));
        else
            up_file_graph.push_back(FileGraph(sub_path, 0, ui_file_graph_neg.index2node));
    }
    ss.clear();
    std::cout << "(IM-Graph)" << std::endl;
    ss << train_im_path;
    while(ss.good())
    {
        getline(ss, sub_path, ',');
        if (im_file_graph.size())
            im_file_graph.push_back(FileGraph(sub_path, 0, im_file_graph.back().index2node));
        else
            im_file_graph.push_back(FileGraph(sub_path, 0, up_file_graph.back().index2node));
    }

    // 1. [Sampler] determine what sampler to be used
    VCSampler ui_sampler(&ui_file_graph);
    VCSampler ui_neg_sampler(&ui_file_graph_neg);
    std::vector<VCSampler> up_sampler;
    for (auto graph: up_file_graph)
        up_sampler.push_back(VCSampler(&graph));
    std::vector<VCSampler> im_sampler;
    for (auto graph: im_file_graph)
        im_sampler.push_back(VCSampler(&graph));

    // 2. [Mapper] define what embedding mapper to be used
    LookupMapper mapper(im_sampler.back().vertex_size, dimension);
    //LookupMapper i_mapper(up_sampler.vertex_size, dimension);

    // 3. [Optimizer] claim the optimizer
    TripletOptimizer optimizer;

    // 4. building the blocks [MF]
    std::cout << "Start Training:" << std::endl;
    unsigned long long total_update_times = (unsigned long long)update_times*1000000;
    unsigned long long worker_update_times = total_update_times/worker;
    unsigned long long finished_update_times = 0;
    Monitor monitor(total_update_times);

    omp_set_num_threads(worker);
    #pragma omp parallel for
    for (int w=0; w<worker; w++)
    {
        int step;
        long user, profile, item, meta;
        std::vector<double> user_embed(dimension, 0.0);
        std::vector<double> item_embed_pos(dimension, 0.0);
        std::vector<double> item_embed_neg(dimension, 0.0);
        std::vector<double> user_loss(dimension, 0.0);
        std::vector<double> item_loss_pos(dimension, 0.0);
        std::vector<double> item_loss_neg(dimension, 0.0);
        std::vector<long> user_collect, pos_collect, neg_collect;
        unsigned long long update=0, report_period = 10000;
        double alpha=init_alpha, alpha_min=alpha*0.0001;

        while (update < worker_update_times)
        {
            // user + profile + item
            user_collect.clear();
            // user
            user_collect.push_back( ui_sampler.draw_a_vertex() );
            // profile
            for (int i=0; i<up_sampler.size(); i++)
            {
                user_collect.push_back( up_sampler[i].draw_a_context_safely(user_collect[0]) );
            }
            // item
            user_collect.push_back( ui_sampler.draw_a_context(user_collect[0]) );
            // embedding
            user_embed = mapper.meta_gcn_embedding(user_collect);

            for (int b=0; b<num_negative; b++)
            {
                pos_collect.clear();
                neg_collect.clear();
                // item + meta
                // item
                pos_collect.push_back( ui_sampler.draw_a_context(user_collect[0]) ); // pos
                neg_collect.push_back( ui_neg_sampler.draw_a_context_uniformly() ); // neg
                // meta
                for (int i=0; i<im_sampler.size(); i++)
                {
                    meta = im_sampler[i].draw_a_context_safely(pos_collect[0]);
                    if (meta!=-1)
                    {
                        pos_collect.push_back(meta);
                    }
                    meta = im_sampler[i].draw_a_context_safely(neg_collect[0]);
                    if (meta!=-1)
                    {
                        neg_collect.push_back(meta);
                    }
                }
                item_embed_pos = mapper.meta_gcn_embedding(pos_collect);
                item_embed_neg = mapper.meta_gcn_embedding(neg_collect);

                // gradient
                optimizer.feed_margin_bpr_loss(user_embed,
                                               item_embed_pos,
                                               item_embed_neg,
                                               8.0,
                                               dimension,
                                               user_loss,
                                               item_loss_pos,
                                               item_loss_neg);
                // update
                for (int i=0; i<pos_collect.size(); i++)
                {
                    mapper.update_with_l2(pos_collect[i], item_loss_pos, alpha, item_reg);
                }
                for (int i=0; i<neg_collect.size(); i++)
                {
                    mapper.update_with_l2(neg_collect[i], item_loss_neg, alpha, item_reg);
                }
                item_loss_pos.assign(dimension, 0.0);
                item_loss_neg.assign(dimension, 0.0);
            }
            // update
            for (int i=0; i<user_collect.size(); i++)
            {
                if (user_collect[i]!=-1)
                    mapper.update_with_l2(user_collect[i], user_loss, alpha, user_reg);
            }
            user_loss.assign(dimension, 0.0);

            // 5. print progress
            update++;
            if (update % report_period == 0) {
                alpha = init_alpha* ( 1.0 - (double)(finished_update_times)/total_update_times );
                if (alpha < alpha_min)
                    alpha = alpha_min;
                finished_update_times += report_period;
                monitor.progress(&finished_update_times);
            }
        }
    }
    monitor.end();
    // get union of item nodes
    std::set<long> item_set;
    for (auto node: ui_file_graph.get_all_to_nodes())
        item_set.insert(node);
    for (auto graph: im_file_graph)
        for (auto node: graph.get_all_from_nodes())
            item_set.insert(node);
    std::vector<long> all_item_nodes(item_set.begin(), item_set.end());
    mapper.save_meta_gcn_to_file(&ui_file_graph, up_file_graph, ui_file_graph.get_all_from_nodes(), save_name_q, 0);
    for (int i=0; i<up_sampler.size(); i++)
        mapper.save_to_file(&up_file_graph[i], up_file_graph[i].get_all_to_nodes(), save_name_k, 0);
    mapper.save_meta_gcn_to_file(&im_file_graph.back(), im_file_graph, all_item_nodes, save_name_q, 1);
    for (int i=0; i<im_sampler.size(); i++)
        mapper.save_to_file(&im_file_graph[i], im_file_graph[i].get_all_to_nodes(), save_name_k, 1);

    return 0;
}
