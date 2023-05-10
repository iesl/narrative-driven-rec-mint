labeled1_poi = "A user likes these recommendations: Mission BBQ in Deptford Township, Keswick Tavern in Glenside, " \
         "Sweet Charlie's in Philadelphia, Mad Mex - Willow Grove in Willow Grove, MOD Pizza in Willow " \
         "Grove, Jules Thin Crust in Jenkintown, Ninja Sushi Hibachi in Philadelphia, Moonlight Diner in " \
         "Glenside, Two Chicks Cafe in New Orleans, Saffron Indian Kitchen in Ambler\n\n" \
         "The user wrote these reviews: This place is always packed! Keswick Tavern is a go to if you're " \
         "looking for great food and large alcohol/beer selection. While, yes there is usually a wait and " \
         "the store is small, I didn't find this very off putting. Happy hour every weekday from 4:30-6:30" \
         " is a must! ( They have such a wide variety of toppings, so no matter you prefer, you can find " \
         "something delicious here! I got the pizza \"flights\" which allow you to try up to four slices at" \
         " a time. They have a ton of options, but there is usually a little bit of a wait for a table. We " \
         "definitely make it our regular spot when we are looking for Sunday morning breakfast! We will " \
         "definitely be coming back for breakfast/brunch again on this trip. One of the best Indian restaurants" \
         " in the area!\n\n" \
         "In response to the request on Reddit: Hi Philadelphia area friends, I will be moving Philadelphia " \
         "soon and I am looking for some popular local restaurants I can turn into my staples without breaking" \
         " the bank! I like a variety of classic American and Asian cuisines - I am partial toward Japanese and " \
         "Indian. I also like novel pizza places though I sometimes feel guilty about how greasy pizzas can get!" \
         " :( I also like to explore menus so I would like places to have large menus. Places which have a generous " \
         "selection of drinks are a plus!"

labeled2_poi = "A user likes these recommendations: Cigar Factory New Orleans in New Orleans, Presta Coffee Roasters in" \
         " Tucson, New Orleans Museum of Art in New Orleans, Sylvain in New Orleans, Central Grocery & Deli in New" \
         " Orleans, Cleo's Mediterranean Cuisine in New Orleans, Waffle House in Treasure Island, Shelter in Reno," \
         " Smugglers Cove Adventure Golf in Madeira Beach, Shaya in New Orleans\n\n" \
         "The user wrote these reviews: The selection is quite excellent and it is fun to watch the rollers make " \
         "the cigars. Tasty coffee, they take their time and it is worth it. Well laid out museum that houses my" \
         " favorite artist Jean Leon Gerome. Great presentation and good combinations of seafood and the salads " \
         "are awesome. The beauty of grabbing the sandwich, some goodies and drinks and having an impromptu picnic" \
         " across the street is too good to pass up. Delicious and super fresh Egyptian Med food. We were there when" \
         " it was pretty dead, but the layout is cool and the bowling lanes are just fine. The staff was nice and" \
         " the holes are not easy. The eggplant dish was superb and the Holy grail of kefta passed with flying" \
         " colors.\n\n" \
         "In response to the request on Reddit: Hello Redditors! I will be driving through New Orleans, Nevada " \
         "and Arizona on a road trip this winter break and staying a few days at each of the spots! I am looking " \
         "for many kinds of recommendations! I like your usual classic American meals and cafes but I'd also love" \
         " to hear about places which show off the local cusine, I am specially excited about the trying the sea " \
         "food in New Orleans! I'm also hoping to visit some of the \"weird\" museums along the way, preference " \
         "for more atrsy. I would also love to hear about fun activities: golf, bowling, kareoke, you name it."

labeled3_poi = "A user likes these recommendations: Ventana Canyon Hiking Trail in Tucson, Thunder Canyon Brewery in " \
         "Tucson, Great Wall China in Tucson, Dao's Tai Pan's in Tucson, US, Sir Veza's Taco Garage in Tucson, US, " \
         "Magpies Gourmet Pizza in Tucson, US, Even Stevens Sandwiches in Tucson, US, The Hut in Tucson, US, Mimi's " \
         "Cafe in Tucson, US\n\n" \
         "The user wrote these reviews: Great view of the city and a handful of flat spots to stop for water and" \
         " to catch your breath. I like the downtown location- the big windows in the front make it a desirable " \
         "and relaxed location. I love to get a pho while my wife dreams of the chicken won ton, and the dumplings. " \
         "Even the simple cheese pizza was delicious and so was the IPA! Love Even Stevens vegan choices for " \
         "breakfast! Relaxed, inexpensive cover, and free shots at the door is a great Way to enter a bar! We were " \
         "meeting for brunch the morning after my wedding the room was prepared well when we arrived.\n\n" \
         "In response to the request on Reddit: Hello r/tucson! I will be visiting your lovely town for my Wedding! " \
         "I would love to hear your recommendations for brunch spots, bars, and breweries in the town - " \
         "I have heard quite a bit about the bar scene in Tucson!. As for food, I typically love Asian food " \
         "and wouldn't be opposed to the occasional pizza or burger spot! I am also looking for recommendations " \
         "for short hiking spots so we can enjoy a quiet hike amidst the hustle and bustle of the wedding!"

narrative_rec_prompt_poi = "User request: Hi Philadelphia area friends, I will be moving Philadelphia soon and I am looking \
for some popular local restaurants I can turn into my staples without breaking the bank! I like a variety of \
classic American and Asian cuisines - I am partial toward Japanese and Indian. I also like novel pizza places \
though I sometimes feel guilty about how greasy pizzas can get! :( I also like to explore menus so I would like \
places to have large menus. Places which have a generous selection of drinks are a plus!\n\n\
10 recommendations for points of interest and their description:\n\
1. Mission BBQ in Deptford Township: This place is always packed! Which is just the side \
effect of having great barbecue, great employees, and a great overall vibe. No matter how \
busy or crowded, every time I have eaten in or got takeout, it never felt like a chore. Staff\
is always friendly and helpful, making the experience great!\n\
2. Keswick Tavern in Glenside: Keswick Tavern is a go to if you're looking for great food and large alcohol/beer\
selection. Their wings are amazing and they have a lot of options for sauces to go with them! Not to mention, \
many specials such trivia on Wednesday's and a DJ after 11 on Friday nights. I've been going regularly for \
dinner and drinks since moving to the area and have yet to be disappointed!\n\
3. Sweet Charlie's in Philadelphia: This place is all about the experience! While, yes there is usually a wait \
and the store is small, I didn't find this very off putting. I heard people complaining about these facts, guess \
you can't make everyone happy. I mean, how bad could it possibly be if you're getting ice cream?!\n\
4. Mad Mex - Willow Grove in Willow Grove: Happy hour every weekday from 4:30-6:30 is a must! (7$ big azz \
margaritas)! They make a darn good margarita too! My friends and I try to come weekly to enjoy dinner and \
margaritas after a long week! There is a great, fun atmosphere and the servers are always friendly too! Super \
casual but the food has yet to disappoint. I always go for an enchilada but I hear the wings (which are half \
price during happy hour) are delicious as well. Will definitely keep coming back for more and recommending to \
friends and family!\n\
5. MOD Pizza in Willow Grove: Great option for easy, fast, yummy and affordable pizza! Not only do they have \
many pre-made options, you can create your own to your liking (always a great option). They have such a wide \
variety of toppings, so no matter you prefer, you can find something delicious here!\n\
6. Jules Thin Crust in Jenkintown: Jules is a great option for healthy, yummy pizza! I got the pizza \"flights\" \
which allow you to try up to four slices at a time.\n\
7. Ninja Sushi Hibachi in Philadelphia: Very good sushi at a really good price! Ninja is a trendy cool \
environment with really delicious sushi rolls. They have a ton of options, but there is usually a little bit \
of a wait for a table. Very much worth it though!\n\
8. Moonlight Diner in Glenside: This is easily the best diner in Glenside! Great and clean atmosphere. The food is\
classic diner food with middle eastern and Mediterranean choices as well.\n\
9. Two Chicks Cafe in New Orleans: We tried this place on our first trip to New Orleans! It was super cute, first \
of all. The service was friendly and the food was amazing! I ordered the traditional Benedict and it was the best \
I have had. We will definitely be coming back for breakfast/brunch again on this trip.\n\
10. Saffron Indian Kitchen in Ambler: One of the best Indian restaurants in the area! I have been multiple times \
and continue to come back for more!"

itemlabeled1_poi = "A user likes this recommendation: Keswick Tavern in Glenside, PA \n The user wrote this review: Keswick \
Tavern is a go to if you're looking for great food and large alcohol/beer selection. Their wings are amazing and they\
 have a lot of options for sauces to go with them! Not to mention, many specials such trivia on Wednesday's and a \
 DJ after 11 on Friday nights. I've been going regularly for dinner and drinks since moving to the area and have \
 yet to be disappointed! \n In response to this request on Reddit: Hello Philadelphia friends, I will be moving to \
 your area for the summer and am looking for restaurant recommendations to turn into my staples. I have recently \
 started to enjoy creative beers and would love a place with a good selection of beers. I also enjoy bars who host\
 regular events which allow socializing with friends or locals in a chilled out environment :) trivia nights, and the like!"

itemlabeled2_poi = "A user likes this recommendation: New Orleans Museum of Art in New Orleans, LA \nThe user wrote this \
review: What a great eclectic selection of sculptures in a beautifully landscaped environment.  There are some old \
favorites (the lipazaner & Hercules that we used to climb on when we were kids & they were in front of the museum). \
The giant tree stump that I got in trouble for trying to sit on when it was inside in the gallery.  Loved the ladder \
to nowhere when it was part of Prospect 1, but it had more power in the devastated lower 9. Great place to wander. \n\
In response to this request on Reddit: r/nola, Please make recommendations of your favorite underappreciated museums! \
I enjoy art museums with a good collections of historical art. I'm dont specially enjoy contemporary art but I would \
be up to checking it out. Or museums which are large enough to house a variety of collections where I can pick and \
choose the collections I enjoy :)"

itemlabeled3_poi = "A user likes this recommendation: Ventana Canyon Hiking Trail in Tucson, AZ \n The user wrote this \
review: Hiked to Window Rock the day before Christmas. This is a long and strenuous hike that took our group of strong \
hikers 7 hours. There is 4310 ft gain in this 12.8 mile \"out-n-back\" hike. Beautiful Sonoran hike with non-stop \"up\" \
to the Window Rock. Will repeat this hike. \n In response to this request on Reddit: Hikers of Tucson! A group of \
colleagues and I am visiting Tucson from San Fransisco for a week long work trip this week and will stay for the \
weekend. We are all big on hiking and specially enjoy a challenging hike. But just as much enjoy a picturesque route. \
Please make your recommendations for trails we should check out!"