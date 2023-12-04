import random
from itertools import chain, combinations
from shapely.geometry import LineString, Point, Polygon



class MACHINE():
    """
        [ MACHINE ]
        MinMax Algorithm을 통해 수를 선택하는 객체.
        - 모든 Machine Turn마다 변수들이 업데이트 됨

        ** To Do **
        MinMax Algorithm을 이용하여 최적의 수를 찾는 알고리즘 생성
           - class 내에 함수를 추가할 수 있음
           - 최종 결과는 find_best_selection을 통해 Line 형태로 도출
               * Line: [(x1, y1), (x2, y2)] -> MACHINE class에서는 x값이 작은 점이 항상 왼쪽에 위치할 필요는 없음 (System이 organize 함)
    """

    def __init__(self, score=[0, 0], drawn_lines=[], whole_lines=[], whole_points=[], location=[]):
        self.id = "MACHINE"
        self.score = [0, 0]  # USER, MACHINE
        self.drawn_lines = []  # Drawn Lines
        self.board_size = 7  # 7 x 7 Matrix
        self.num_dots = 0
        self.whole_points = []
        self.location = []
        self.triangles = []  # [(a, b), (c, d), (e, f)]
        self.count_turn = 0
        self.evaluate_score = [0,0]

    # def find_best_selection(self):
    #     available = [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if
    #                  self.check_availability([point1, point2])]
    #     return random.choice(available)

    def find_best_selection(self):
        print("start", self.triangles)
        self.count_turn += 1
        if(self.count_turn < 1):
            print("selection begins")
            # count how many times each point has been used to draw lines so far
            points_to_drawn_times = { point: 0 for point in self.whole_points}

            # see the list of connected points to each point
            graph = { point: [] for point in self.whole_points }

            for (point1, point2) in self.drawn_lines:
                points_to_drawn_times[point1] += 1
                points_to_drawn_times[point2] += 1
                graph[point1].append((point1, point2))
                graph[point2].append((point1, point2))

            # reverse the key-value relation
            drawn_times_to_points = { points_to_drawn_times[point]: [] for point in points_to_drawn_times }
            for point in points_to_drawn_times:
                drawn_times_to_points[points_to_drawn_times[point]].append(point)
            
        
            # draw traingle if possible
            count_maximum_drawn__times = max(list(drawn_times_to_points.keys()))

            available = []
            
            available = self.find_lines_generating_two_triangles(graph)
            if len(available) > 0:
                return random.choice(available)        

            for drawn_times in range(count_maximum_drawn__times, 1, -1):
                if drawn_times in drawn_times_to_points:
                    points = drawn_times_to_points[drawn_times]
                    for point in points:
                        lines = graph[point]
                        available = available + self.find_best_triangle_lines(lines, graph)
            
            


            if len(available) > 0:
                print("draw a traingle")
                return random.choice(available)

            # extract points that have not been used to draw any line yet
            points_not_drawn = drawn_times_to_points[0] if 0 in drawn_times_to_points else []

            # select 2 points that hasn't been used to draw a line
            if len(points_not_drawn) >= 2:
                print("draw using unused points")
                available = self.find_available(points_not_drawn)
            
            # draw lines unfavorable to opponent
            lines_unfavorable_to_opponent = []
            for line in available:
                if not self.can_make_triangle(line, graph):
                    lines_unfavorable_to_opponent = lines_unfavorable_to_opponent + [line]
            
            if len(lines_unfavorable_to_opponent) > 0:
                return random.choice(lines_unfavorable_to_opponent)

            # pick among all the possible cases
            if len(available) == 0:
                available = self.find_available(self.whole_points)

            return random.choice(available)
        else:
            graph = { point: [] for point in self.whole_points }

            for (point1, point2) in self.drawn_lines:
                graph[point1].append((point1, point2))
                graph[point2].append((point1, point2))

            best_score = float('-inf')
            best_move = None
            cache = {}
            for move in self.get_all_possible_moves():
                self.make_move(move, True, graph)
                score = self.minmax(3, float('-inf'), float('inf'), False, cache, graph)
                # print('=== score 출력 ===')
                # print(score)
                self.undo_move(move, True, graph)
                if score > best_score:
                    best_score = score
                    best_move = move
                if best_move is None:
                    available = [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if
                        self.check_availability([point1, point2])]
                    return random.choice(available)
            print('drawn lines', self.drawn_lines)
            print('graph', graph)
            return best_move

    def check_availability(self, line):
        line_string = LineString(line)

        # Must be one of the whole points
        condition1 = (line[0] in self.whole_points) and (line[1] in self.whole_points)

        # Must not skip a dot
        condition2 = True
        for point in self.whole_points:
            if point == line[0] or point == line[1]:
                continue
            else:
                if bool(line_string.intersection(Point(point))):
                    condition2 = False

        # Must not cross another line
        condition3 = True
        for l in self.drawn_lines:
            if len(list(set([line[0], line[1], l[0], l[1]]))) == 3:
                continue
            elif bool(line_string.intersection(LineString(l))):
                condition3 = False

        # Must be a new line
        condition4 = (line not in self.drawn_lines)

        if condition1 and condition2 and condition3 and condition4:
            return True
        else:
            return False

    def evaluate(self):
        # Higher score if more triangles can be formed.
        # Minus score if the opponent can form triangles.
        user_score, machine_score = self.evaluate_score
        # print()
        # print('=== evaluate 결과 ===')
        # print(machine_score - user_score)
        return machine_score - user_score
    
    def retrieve_cache_key(self, maximizingPlayer):
        tuple_lines = tuple(tuple(line) for line in self.drawn_lines)
        cache_key = (tuple(sorted(tuple_lines)), maximizingPlayer)
        return cache_key


    # Organization Functions
    def organize_points(self, point_list):
        return sorted(point_list, key=lambda x: (x[0], x[1]))
        # return point_list


    def minmax(self, depth, alpha, beta, maximizingPlayer, cache, graph):
        # print('=== minmax 함수 진입 ===')
        # print('=== depth 출력 ===')
        # print(depth)
        cache_key = self.retrieve_cache_key(maximizingPlayer)
        if cache_key in cache:
            return cache[cache_key]

        if depth == 0 or self.check_endgame():
            # print(self.drawn_lines)
            cache[cache_key] = self.evaluate()
            return cache[cache_key]

        if maximizingPlayer:
            maxEval = float('-inf')
            for move in self.get_all_possible_moves():
                self.make_move(move, maximizingPlayer, graph)
                evaluation = self.minmax(depth - 1, alpha, beta, False, cache, graph)
                self.undo_move(move, maximizingPlayer, graph)
                maxEval = max(maxEval, evaluation)
                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    # print("alpha beta cutoff occured", self.drawn_lines)
                    break
            cache[cache_key] = maxEval
            return cache[cache_key]
        else:
            minEval = float('inf')
            for move in self.get_all_possible_moves():
                self.make_move(move, maximizingPlayer, graph)
                evaluation = self.minmax(depth - 1, alpha, beta, True, cache, graph)
                self.undo_move(move, maximizingPlayer, graph)
                minEval = min(minEval, evaluation)
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break
            cache[cache_key] = minEval
            return cache[cache_key]

    def get_all_possible_moves(self):
        # Start with all possible combinations of points for lines
        all_combinations = combinations(self.whole_points, 2)

        # Filter out combinations that are not valid moves
        possible_moves = [
            self.organize_points(move) for move in all_combinations if self.check_availability(move)
        ]

        #print("get_all_possible_moves: ", possible_moves)
        return possible_moves

    def undo_move(self, move, maximizingPlayer, graph):
        point1, point2 = move
        print("graph")
        print(graph[point1], graph[point2])
        print((point1, point2,))
        graph[point1].remove((point1, point2))
        graph[point2].remove((point1, point2))
        # Remove the move from the list of drawn lines
        if move in self.drawn_lines:
            self.drawn_lines.remove(move)
        # Recalculate the score since it may have changed by removing this line
        self.recalculate_score_after_undo(move, maximizingPlayer)

    def recalculate_score_after_undo(self, move, maximizingPlayer):
        # Remove any triangles from the score that included this line
        # This assumes that you have a list of triangles and their corresponding lines

        triangles_to_remove = [triangle for triangle in self.triangles if move[0] in triangle and move[1] in triangle]

        
        for triangle in triangles_to_remove:
            # If the triangle was contributing to the MACHINE's score, decrease the score
            if maximizingPlayer:
                self.evaluate_score[1] -= 1  # Decrease MACHINE's score
            else:
                self.evaluate_score[0] -= 1  # Increase USER's score
            # Remove the triangle from the list of completed triangles
            self.triangles.remove(triangle)

    # def find_triangles(self, lines):
    #     # List to hold the triangles found
    #     found_triangles = []
    #
    #     # Check every combination of three lines to see if they form a triangle
    #     for combo in combinations(lines, 3):
    #         if self.do_lines_form_triangle(combo):
    #             # If they do form a triangle, get the triangle's points
    #             triangle_points = self.get_triangle_points(combo)
    #             if triangle_points and self.is_triangle_empty(triangle_points):
    #                 # If the triangle is valid and empty, add it to the list
    #                 found_triangles.append(triangle_points)
    #
    #     return found_triangles

    def find_triangles(self, lines, player_id):
        # This will store the triangles found
        triangles = []

        # Look at all combinations of three lines to find triangles
        for line1, line2, line3 in combinations(lines, 3):
            if self.do_lines_form_triangle((line1, line2, line3)):
                # If these lines form a triangle, find the points of the triangle
                triangle_points = self.get_triangle_points((line1, line2, line3))
                if triangle_points:
                    # If valid points are returned, create a triangle object (or any structure you prefer)
                    triangle = {
                        'points': triangle_points,
                        'lines': (line1, line2, line3),
                        'player_id': player_id
                    }
                    # Check if the triangle is not already found to avoid duplicates
                    if triangle not in triangles:
                        triangles.append(triangle)
        return triangles

    def is_game_over(self):
        # Check if all possible lines have been drawn
        if len(self.drawn_lines) == len(list(combinations(self.whole_points, 2))):
            return True

        # If the maximum score is reached by either player, the game is over
        max_score_possible = len(list(combinations(self.whole_points, 3)))  # This would be the max number of triangles
        if max(self.score) >= max_score_possible:
            return True

        # Check if there are no possible moves that can lead to a new triangle
        for move in self.get_all_possible_moves():
            if any(self.does_line_complete_triangle(move, player_id) for player_id in [self.id, "USER"]):
                return False

        # If none of the conditions are met, the game is not over
        return True

    def does_line_complete_triangle(self, move, player_id):
        # Assume a temporary move is made
        self.drawn_lines.append(move)

        # Check if this move with the existing lines of the player forms a triangle
        triangles = self.find_triangles(self.drawn_lines, player_id)

        # Undo the temporary move
        self.drawn_lines.remove(move)

        # If any triangle is formed with the move, it can potentially complete a triangle
        return bool(triangles)

    def make_move(self, move, maximizingPlayer, graph):
        print(self.drawn_lines)
        # Add the move to the list of drawn lines
        self.drawn_lines.append(move)

        point1, point2 = move
        graph[point1].append((point1, point2,))
        graph[point2].append((point1, point2,))


        # Update the score if this move completes a triangle
        # The `update_score` function would need to be implemented to check for this
        self.update_score(move, maximizingPlayer, graph)

        # Add any other game state updates related to making a move here
        # For example, if you have a current player indicator, switch it to the other player

    def update_score(self, move, maximizingPlayer, graph):
        # Check if the new move completes any new triangles
        new_triangles = self.find_new_triangles(move, graph)

        # Update the score for each new triangle
        for triangle in new_triangles:
            # Assuming the score is indexed with 0 for USER and 1 for MACHINE
            if maximizingPlayer:
                self.evaluate_score[1] += 1  # Increment MACHINE's score
            else:
                self.evaluate_score[0] += 1  # Increment USER's score
            self.triangles.append(triangle)  # Add the triangle to the list of completed triangles

    def find_new_triangles(self, new_line, graph):
        # Find all sets of two lines from the existing lines that, together with the new line, could form a triangle.
        print(graph)
        possible_triangles = [
            (line1, line2, new_line)
            for line1 in graph[new_line[0]]
            for line2 in graph[new_line[1]]
            if line1 != line2 and line1 != new_line and line2 != new_line
        ]

        # Filter out sets that do not form a triangle or form a triangle with points inside
        new_triangles = []
        for lines in possible_triangles:
            if self.do_lines_form_triangle(lines):
                triangle_points = self.get_triangle_points(lines)
                if triangle_points and self.is_triangle_valid(triangle_points):
                    new_triangles.append(triangle_points)

        return new_triangles

    def do_lines_form_triangle(self, lines):
        # Extract all points from the lines
        points = [point for line in lines for point in line]

        # A triangle is formed if there are exactly 3 unique points
        unique_points = set(points)
        if len(unique_points) == 3:
            # Ensure each unique point is an endpoint for exactly two lines
            return all(points.count(point) == 2 for point in unique_points)
        return False

    def get_triangle_points(self, lines):
        # Flatten the list of lines into a list of points
        all_points = [point for line in lines for point in line]

        # Use a set to eliminate duplicate points
        unique_points = list(set(all_points))

        # Since we've previously determined these lines form a triangle,
        # there should be exactly 3 unique points.
        if len(unique_points) == 3:
            return unique_points
        else:
            # If there are not exactly 3 unique points, something has gone wrong.
            return None

    def is_triangle_empty(self, triangle_points):
        # Construct a Polygon object from the triangle vertices
        triangle = Polygon(triangle_points)

        # Check each point to see if it is inside the triangle
        for point in self.whole_points:
            if Point(point).within(triangle):
                # If any point is inside the triangle, it's not empty
                return False

        # If no points are inside the triangle, it is empty
        return True

    def is_triangle_valid(self, triangle):
        # Check if the triangle does not contain any other points inside, including on the edges
        # This is based on the rules described in the Gaining Territory game
        for point in self.whole_points:
            if point in triangle:
                continue  # Skip the vertices of the triangle
            if self.is_point_inside_triangle(point, list(combinations(triangle, 2))):
                return False
        return True

    def is_point_inside_triangle2(self, point, triangle):
        (x1, y1), (x2, y2), (x3, y3) = triangle
        xp, yp = point

        # Calculate the barycentric coordinates
        denom = ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
        a = ((y2 - y3) * (xp - x3) + (x3 - x2) * (yp - y3)) / denom
        b = ((y3 - y1) * (xp - x3) + (x1 - x3) * (yp - y3)) / denom
        c = 1 - a - b

        # Check if point is inside the triangle
        return 0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1

    def check_endgame(self):
        remain_to_draw = [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if
                          self.check_availability([point1, point2])]
        return False if remain_to_draw else True




# 추가된 코드
    def find_available(self, points):
        return [[point1, point2] for (point1, point2) in list(combinations(points, 2)) if
                self.check_availability([point1, point2])]

        # todo: draw some line when user tries to fill the entire inner lines

    def find_best_triangle_lines(self, lines, graph):
        print("lines")
        print(lines)
        triangle_lines = []
        for [(point1, point2), (point3, point4)] in list(combinations(lines, 2)):
            print([(point1, point2), (point3, point4)])
            if self.are_points_same(point1, point3):
                if self.check_availability([point2, point4]):
                    if self.count_points_inside_triangle(
                            [(point1, point2), (point3, point4), (point2, point4)]) == 0 and len(
                            self.find_lines_connecting_two_triangles([point1, point2, point4], graph)) == 0:
                        triangle_lines.append([point2, point4])
            elif self.are_points_same(point1, point4):
                if self.check_availability([point2, point3]):
                    if self.count_points_inside_triangle(
                            [(point1, point2), (point3, point4), (point2, point3)]) == 0 and len(
                            self.find_lines_connecting_two_triangles([point1, point2, point3], graph)) == 0:
                        triangle_lines.append([point2, point3])
            elif self.are_points_same(point2, point3):
                if self.check_availability([point1, point4]):
                    if self.count_points_inside_triangle(
                            [(point1, point2), (point3, point4), (point1, point4)]) == 0 and len(
                            self.find_lines_connecting_two_triangles([point1, point2, point4], graph)) == 0:
                        triangle_lines.append([point1, point4])
            else:  # point 2 == point4
                if self.check_availability([point1, point3]):
                    if self.count_points_inside_triangle(
                            [(point1, point2), (point3, point4), (point1, point3)]) == 0 and len(
                            self.find_lines_connecting_two_triangles([point1, point2, point3], graph)) == 0:
                        triangle_lines.append([point1, point3])
        print("triangle_lines")
        print(triangle_lines)
        return triangle_lines

    def count_points_inside_triangle(self, lines: list):
        triangle = self.organize_points(list(set(chain(*[lines[0], lines[1], lines[2]]))))
        count_points = 0
        for point in self.whole_points:
            if point in triangle:
                continue
            if self.is_point_inside_triangle(point, lines):
                count_points += 1
        print("number of points inside triangle")
        print(count_points)
        return count_points

    def is_point_inside_triangle(self, point: list, lines: list) -> bool:
        
        
        triangle = self.organize_points(list(set(chain(*[lines[0], lines[1], lines[2]]))))
        return bool(Polygon(triangle).intersection(Point(point)))

    def is_vertex_of_triangle(self, point: list, triangle: list) -> bool:
        return point in triangle

    def are_two_vertices_same(self, triangle1, triangle2):
        count_same_vertices = 0
        for point1 in triangle1:
            for point2 in triangle2:
                if point1[0] == point2[0] and point1[1] == point2[1]:
                    count_same_vertices += 1

        return count_same_vertices == 2

    def find_different_vertices(self, triangle1, triangle2):
        different_vertices = []
        points = triangle1 + triangle2
        for point in points:
            if point not in triangle1 or point not in triangle2:
                different_vertices.append(point)
        return different_vertices

    def find_lines_generating_two_triangles(self, graph):
        # find a line when the two triangles have the same two
        triangle_lines = []
        for triangle in self.triangles:
            triangle_lines = triangle_lines + self.find_lines_connecting_two_triangles(triangle, graph)
        if len(triangle_lines) == 0:
            triangle_lines = self.find_lines_with_point_on_line()

        return triangle_lines
    
    def can_make_triangle(self, new_line, graph):
        [point1, point2] = new_line

        lines_of_point1 = graph[point1]
    
        for [previous_point1, previous_point2] in lines_of_point1:
            if self.are_points_same(point1, previous_point1) and not self.are_points_same(point2, previous_point2):
                if self.check_availability([point2, previous_point2]):
                    return True
            elif self.are_points_same(point1, previous_point2) and not self.are_points_same(point2, previous_point1):
                if self.check_availability([point2, previous_point1]):
                    return True
        
        lines_of_point2 = graph[point2]
    
        for [previous_point1, previous_point2] in lines_of_point2:
            if self.are_points_same(point2, previous_point1) and not self.are_points_same(point1, previous_point2):
                if self.check_availability([point1, previous_point2]):
                    return True
            elif self.are_points_same(point2, previous_point2) and not self.are_points_same(point1, previous_point1):
                if self.check_availability([point1, previous_point1]):
                    return True
        
        return False

    def find_lines_connecting_two_triangles(self, triangle, graph):
        triangle_lines = []

        print(triangle)
        for [vertex_in_triangle1, vertex_in_triangle2] in list(combinations(triangle, 2)):
            lines1 = graph[vertex_in_triangle1]
            lines2 = graph[vertex_in_triangle2]
            for [point1, point2] in lines1:
                for [point3, point4] in lines2:
                    print("point1, point2 point3 point4")
                    print(point1, point2, point3, point4)
                    if self.are_points_same(point1, point3) and point1 not in triangle:
                        [vertex] = [vertex for vertex in triangle if vertex not in [point1, point2, point3, point4]]
                        if self.check_availability([vertex, point1]):
                            triangle_lines.append([vertex, point1])
                    elif self.are_points_same(point1, point4) and point1 not in triangle:
                        [vertex] = [vertex for vertex in triangle if vertex not in [point1, point2, point3, point4]]
                        if self.check_availability([vertex, point1]):
                            triangle_lines.append([vertex, point1])
                    elif self.are_points_same(point2, point3) and point2 not in triangle:
                        [vertex] = [vertex for vertex in triangle if vertex not in [point1, point2, point3, point4]]
                        if self.check_availability([vertex, point2]):
                            triangle_lines.append([vertex, point2])
                    elif self.are_points_same(point2, point4) and point2 not in triangle:
                        [vertex] = [vertex for vertex in triangle if vertex not in [point1, point2, point3, point4]]
                        if self.check_availability([vertex, point2]):
                            triangle_lines.append([vertex, point2])
        return triangle_lines

    def find_lines_with_point_on_line(self):
        triangle_lines = []
        # find a line when a point is in the middle of line
        for point in self.whole_points:
            for triangle in self.triangles:
                for line in list(combinations(triangle, 2)):
                    if self.is_point_on_line(point, line):
                        [left_vertex] = [vertex for vertex in triangle if vertex not in line]
                        if self.check_availability([point, left_vertex]):
                            triangle_lines.append([point, left_vertex])

        return triangle_lines

    def is_point_on_line(self, point, line):
        bool(LineString(line).intersects(Point(point)))

    def are_points_same(self, point1, point2):
        return point1[0] == point2[0] and point1[1] == point2[1]

        